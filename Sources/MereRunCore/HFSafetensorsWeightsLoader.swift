import Foundation
import MLX
import MLXNN

public enum HFSafetensorsWeightsLoader {
    private static let residualEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_SVD_RESIDUAL"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    private static let residualDebug: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_SVD_RESIDUAL_DEBUG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    private static let verifyUnusedKeys: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_VERIFY_UNUSED_KEYS"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    public enum LoaderError: LocalizedError {
        case indexFileMissing(URL)
        case shardFileMissing(URL)
        case notQuantizedWeights
        case missingQuantizedWeight(String)

        public var errorDescription: String? {
            switch self {
            case .indexFileMissing(let url):
                return "Safetensors index file not found: \(url.path)"
            case .shardFileMissing(let url):
                return "Safetensors shard file not found: \(url.path)"
            case .notQuantizedWeights:
                return "Expected quantized weights but none found (missing .scales keys)"
            case .missingQuantizedWeight(let key):
                return "Missing quantized weight for key: \(key)"
            }
        }
    }

    public struct ShardProgress: Sendable, Hashable {
        public let shardIndex: Int
        public let shardCount: Int
        public let shardFilename: String

        public init(shardIndex: Int, shardCount: Int, shardFilename: String) {
            self.shardIndex = shardIndex
            self.shardCount = shardCount
            self.shardFilename = shardFilename
        }
    }

    public typealias QuantizedModuleResolver = (
        _ path: String,
        _ module: Module,
        _ quantizedWeight: MLXArray,
        _ scales: MLXArray,
        _ biases: MLXArray?,
        _ fallbackGroupSize: Int,
        _ fallbackBits: Int
    ) -> (groupSize: Int, bits: Int, mode: QuantizationMode)

    /// Applies weights from a Hugging Face sharded safetensors index (`*.safetensors.index.json`).
    ///
    /// This streams shards one-at-a-time to keep peak memory low.
    public static func applyShardedWeights(
        indexURL: URL,
        to model: Module,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .none,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] },
        progressHandler: (@Sendable (ShardProgress) -> Void)? = nil
    ) throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: indexURL.path) else {
            throw LoaderError.indexFileMissing(indexURL)
        }

        let indexData = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: indexData)

        let baseURL = indexURL.deletingLastPathComponent()
        let shards = index.shardFilenames
        for (shardIndex, shardFilename) in shards.enumerated() {
            let shardURL = baseURL.appending(path: shardFilename)
            guard fm.fileExists(atPath: shardURL.path) else {
                throw LoaderError.shardFileMissing(shardURL)
            }

            try applyWeights(
                url: shardURL,
                to: model,
                dtype: dtype,
                verify: verify,
                mapper: { key, value in
                    guard index.weightMap[key] == shardFilename else { return [] }
                    return mapper(key, value)
                }
            )

            progressHandler?(ShardProgress(
                shardIndex: shardIndex,
                shardCount: shards.count,
                shardFilename: shardFilename
            ))
        }
    }

    /// Applies weights from a single safetensors file (`*.safetensors`).
    public static func applyWeights(
        url: URL,
        to model: Module,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .none,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] }
    ) throws {
        let arrays = try MLX.loadArrays(url: url)
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(arrays.count)

        for (key, rawValue) in arrays {
            let value = castIfNeeded(rawValue, dtype: dtype)
            updates.append(contentsOf: mapper(key, value))
        }

        if updates.isEmpty {
            return
        }

        try model.update(parameters: ModuleParameters.unflattened(updates), verify: verify)
    }

    public static func castIfNeeded(_ value: MLXArray, dtype: DType?) -> MLXArray {
        guard let dtype, value.dtype != dtype else { return value }
        guard value.dtype.isFloatingPoint else { return value }
        return value.asType(dtype)
    }

    /// Common weight layout conversion for convolution kernels saved by PyTorch (`[out, in, kH, kW]`).
    public static func convWeightOIHWToOHWI(_ value: MLXArray) -> MLXArray {
        guard value.ndim == 4 else { return value }
        let transposed = value.transposed(0, 2, 3, 1)
        // Ensure a contiguous buffer after transpose (matches `mlx-swift-examples` load pattern).
        return transposed.reshaped(-1).reshaped(transposed.shape)
    }

    // MARK: - Quantized Weight Loading

    /// Loads all arrays from a sharded safetensors index into a dictionary.
    public static func loadShardedArrays(
        indexURL: URL,
        progressHandler: (@Sendable (ShardProgress) -> Void)? = nil
    ) throws -> [String: MLXArray] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: indexURL.path) else {
            throw LoaderError.indexFileMissing(indexURL)
        }

        let indexData = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: indexData)

        let baseURL = indexURL.deletingLastPathComponent()
        let shards = index.shardFilenames

        var allArrays: [String: MLXArray] = [:]
        allArrays.reserveCapacity(index.weightMap.count)

        for (shardIndex, shardFilename) in shards.enumerated() {
            let shardURL = baseURL.appending(path: shardFilename)
            guard fm.fileExists(atPath: shardURL.path) else {
                throw LoaderError.shardFileMissing(shardURL)
            }

            let arrays = try MLX.loadArrays(url: shardURL)
            for (key, value) in arrays where index.weightMap[key] == shardFilename {
                allArrays[key] = value
            }

            progressHandler?(ShardProgress(
                shardIndex: shardIndex,
                shardCount: shards.count,
                shardFilename: shardFilename
            ))
        }

        return allArrays
    }

    /// Checks if a weight dictionary contains quantized weights (has `.scales` keys).
    public static func isQuantized(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { $0.hasSuffix(".scales") }
    }

    private static func resolveQuantizationParams(
        for module: Module,
        quantizedWeight: MLXArray,
        scales: MLXArray,
        fallbackGroupSize: Int,
        fallbackBits: Int
    ) -> (groupSize: Int, bits: Int) {
        // The quantization command currently doesn't persist bits/groupSize as explicit metadata.
        // Infer them from tensor shapes:
        // - Quantized weights pack 32/bits values per element in the last dimension.
        // - Scales last dimension is typically (inDim / groupSize).

        let inDim: Int? = {
            if let linear = module as? Linear {
                return linear.shape.1
            }
            if let embedding = module as? Embedding, embedding.weight.ndim == 2 {
                return embedding.weight.shape2.1
            }
            return nil
        }()

        var resolvedBits = fallbackBits
        if let inDim, inDim > 0, quantizedWeight.ndim == 2 {
            let packedInDim = quantizedWeight.shape2.1
            let numerator = packedInDim * 32
            if numerator % inDim == 0 {
                let inferredBits = numerator / inDim
                if (2...8).contains(inferredBits) {
                    resolvedBits = inferredBits
                }
            }
        }

        var resolvedGroupSize = fallbackGroupSize
        if let inDim, inDim > 0, scales.ndim == 2 {
            let scaleCols = scales.shape2.1
            if scaleCols > 0, inDim % scaleCols == 0 {
                resolvedGroupSize = inDim / scaleCols
            }
        }

        return (resolvedGroupSize, resolvedBits)
    }

    /// Applies pre-quantized weights to a model, replacing Linear layers with QuantizedLinear.
    ///
    /// This function:
    /// 1. Loads all weight arrays from sharded safetensors
    /// 2. Identifies Linear layers that have corresponding `.scales` keys
    /// 3. Creates QuantizedLinear replacements with the pre-quantized weights
    /// 4. Applies remaining non-quantized parameters normally
    ///
    /// - Parameters:
    ///   - indexURL: URL to the `model.safetensors.index.json` file
    ///   - model: The model to load weights into
    ///   - groupSize: Quantization group size (must match how weights were quantized)
    ///   - bits: Bits per weight (must match how weights were quantized)
    ///   - keyMapper: Optional function to transform weight keys before matching
    ///   - progressHandler: Optional callback for shard loading progress
    public static func applyQuantizedWeights(
        indexURL: URL,
        to model: Module,
        groupSize: Int = 64,
        bits: Int = 4,
        applySVDResiduals: Bool? = nil,
        quantizedModuleResolver: QuantizedModuleResolver? = nil,
        keyMapper: ((String) -> String)? = nil,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] },
        progressHandler: (@Sendable (ShardProgress) -> Void)? = nil
    ) throws {
        let useResiduals = applySVDResiduals ?? Self.residualEnabled

        // 1. Load all weight arrays
        var weights = try loadShardedArrays(indexURL: indexURL, progressHandler: progressHandler)

        // Apply key mapping if provided
        if let keyMapper {
            var mappedWeights: [String: MLXArray] = [:]
            mappedWeights.reserveCapacity(weights.count)
            for (key, value) in weights {
                mappedWeights[keyMapper(key)] = value
            }
            weights = mappedWeights
        }

        // 2. Check if quantized
        guard isQuantized(weights) else {
            throw LoaderError.notQuantizedWeights
        }

        // 3. Build module replacements and parameter updates
        let leafModules = model.leafModules().flattened()
        var quantizedReplacements: [String: Module] = [:]
        var usedKeys: Set<String> = []

        var residualApplied = 0
        for (path, module) in leafModules {
            // Check if this module has quantized weights (has .scales key)
            let scalesKey = "\(path).scales"
            guard let scales = weights[scalesKey] else { continue }

            let weightKey = "\(path).weight"
            let biasesKey = "\(path).biases"  // quantization biases
            let svdUpKey = "\(path).svd_up"
            let svdDownKey = "\(path).svd_down"

            guard let qWeight = weights[weightKey] else {
                throw LoaderError.missingQuantizedWeight(weightKey)
            }

            let qBiases = weights[biasesKey]
            let svdUp = weights[svdUpKey]
            let svdDown = weights[svdDownKey]
            if svdUp != nil { usedKeys.insert(svdUpKey) }
            if svdDown != nil { usedKeys.insert(svdDownKey) }

            if module is Linear {
                // Quantized Linear layer
                let biasKey = "\(path).bias"
                let linearBias = weights[biasKey]

                let qp = resolveQuantizationParams(
                    for: module,
                    quantizedWeight: qWeight,
                    scales: scales,
                    fallbackGroupSize: groupSize,
                    fallbackBits: bits
                )
                let resolved = quantizedModuleResolver?(
                    path,
                    module,
                    qWeight,
                    scales,
                    qBiases,
                    qp.groupSize,
                    qp.bits
                ) ?? (groupSize: qp.groupSize, bits: qp.bits, mode: QuantizationMode.affine)

                let quantized: Module
                if useResiduals, let svdUp, let svdDown {
                    quantized = ResidualQuantizedLinear(
                        weight: qWeight,
                        bias: linearBias,
                        scales: scales,
                        biases: qBiases,
                        groupSize: resolved.groupSize,
                        bits: resolved.bits,
                        mode: resolved.mode,
                        residualDown: svdDown,
                        residualUp: svdUp
                    )
                    residualApplied += 1
                } else {
                    quantized = PortableQuantizedLinear(
                        weight: qWeight,
                        bias: linearBias,
                        scales: scales,
                        biases: qBiases,
                        groupSize: resolved.groupSize,
                        bits: resolved.bits,
                        mode: resolved.mode
                    )
                }
                quantizedReplacements[path] = quantized

                if linearBias != nil { usedKeys.insert(biasKey) }
            } else if module is Embedding {
                // Quantized Embedding layer - use our custom class for pre-quantized weights
                let qp = resolveQuantizationParams(
                    for: module,
                    quantizedWeight: qWeight,
                    scales: scales,
                    fallbackGroupSize: groupSize,
                    fallbackBits: bits
                )
                let resolved = quantizedModuleResolver?(
                    path,
                    module,
                    qWeight,
                    scales,
                    qBiases,
                    qp.groupSize,
                    qp.bits
                ) ?? (groupSize: qp.groupSize, bits: qp.bits, mode: QuantizationMode.affine)
                let quantized = PreQuantizedEmbedding(
                    weight: qWeight,
                    scales: scales,
                    biases: qBiases,
                    groupSize: resolved.groupSize,
                    bits: resolved.bits,
                    mode: resolved.mode
                )
                quantizedReplacements[path] = quantized
            } else {
                // Unknown module type with quantized weights - skip (will be loaded as remaining params)
                continue
            }

            usedKeys.insert(weightKey)
            usedKeys.insert(scalesKey)
            if qBiases != nil { usedKeys.insert(biasesKey) }
        }

        // 4. Apply module replacements (Linear → QuantizedLinear)
        // MLX's update(modules:) requires complete array structures when updating array elements.
        // We must provide all array elements, not just the ones being replaced.
        if !quantizedReplacements.isEmpty {
            // Group replacements by their array parent path
            var arrayUpdates: [String: [(Int, Module)]] = [:]
            var directUpdates: [(String, Module)] = []

            for (path, replacement) in quantizedReplacements {
                // Check if this path ends with a numeric index (e.g., "cap_embedder.1")
                let components = path.split(separator: ".")
                if let lastComponent = components.last, let index = Int(lastComponent) {
                    // This is inside an array/tuple - group by parent path
                    let parentPath = components.dropLast().joined(separator: ".")
                    arrayUpdates[parentPath, default: []].append((index, replacement))
                } else {
                    // Not in an array - update directly
                    directUpdates.append((path, replacement))
                }
            }

            // Build complete array updates by including all original modules
            var moduleUpdates: [(String, Module)] = directUpdates
            for (parentPath, indexedReplacements) in arrayUpdates {
                // Get all current modules in this array
                let currentModules = leafModules.filter { (p, _) in
                    let pComponents = p.split(separator: ".")
                    if pComponents.count < 2 { return false }
                    let pParent = pComponents.dropLast().joined(separator: ".")
                    return pParent == parentPath && Int(pComponents.last!) != nil
                }

                // Build replacement map
                var replacementMap: [Int: Module] = [:]
                for (index, replacement) in indexedReplacements {
                    replacementMap[index] = replacement
                }

                // Build complete array with all modules
                for (modulePath, originalModule) in currentModules {
                    let index = Int(modulePath.split(separator: ".").last!)!
                    let module = replacementMap[index] ?? originalModule
                    moduleUpdates.append((modulePath, module))
                }
            }

            if !moduleUpdates.isEmpty {
                model.update(modules: ModuleChildren.unflattened(moduleUpdates))
            }
        }

        // 5. Apply remaining parameters (non-quantized weights like norms, embeddings, etc.)
        var remainingUpdates: [(String, MLXArray)] = []
        for (key, value) in weights where !usedKeys.contains(key) {
            remainingUpdates.append(contentsOf: mapper(key, value))
        }

        if !remainingUpdates.isEmpty {
            let verify: Module.VerifyUpdate = Self.verifyUnusedKeys ? .noUnusedKeys : .none
            try model.update(parameters: ModuleParameters.unflattened(remainingUpdates), verify: verify)
        }

        if residualApplied > 0, Self.residualDebug {
            let message = "[HFSafetensorsWeightsLoader] Applied SVD residuals to \(residualApplied) layers\n"
            FileHandle.standardError.write(Data(message.utf8))
        }
    }

    /// Applies pre-quantized weights from a dictionary to a model.
    /// Use this when weights are already loaded into memory.
    public static func applyQuantizedWeightsFromArrays(
        _ weights: [String: MLXArray],
        to model: Module,
        groupSize: Int = 64,
        bits: Int = 4,
        applySVDResiduals: Bool? = nil,
        quantizedModuleResolver: QuantizedModuleResolver? = nil,
        keyMapper: ((String) -> String)? = nil,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] }
    ) throws {
        let useResiduals = applySVDResiduals ?? Self.residualEnabled

        var mappedWeights = weights

        // Apply key mapping if provided
        if let keyMapper {
            var newWeights: [String: MLXArray] = [:]
            newWeights.reserveCapacity(weights.count)
            for (key, value) in weights {
                newWeights[keyMapper(key)] = value
            }
            mappedWeights = newWeights
        }

        // Check if quantized
        guard isQuantized(mappedWeights) else {
            throw LoaderError.notQuantizedWeights
        }

        // Build module replacements and parameter updates
        let leafModules = model.leafModules().flattened()
        var quantizedReplacements: [String: Module] = [:]
        var usedKeys: Set<String> = []

        var residualApplied = 0
        for (path, module) in leafModules {
            let scalesKey = "\(path).scales"
            guard let scales = mappedWeights[scalesKey] else { continue }

            let weightKey = "\(path).weight"
            let biasesKey = "\(path).biases"
            let svdUpKey = "\(path).svd_up"
            let svdDownKey = "\(path).svd_down"

            guard let qWeight = mappedWeights[weightKey] else {
                throw LoaderError.missingQuantizedWeight(weightKey)
            }

            let qBiases = mappedWeights[biasesKey]
            let svdUp = mappedWeights[svdUpKey]
            let svdDown = mappedWeights[svdDownKey]
            if svdUp != nil { usedKeys.insert(svdUpKey) }
            if svdDown != nil { usedKeys.insert(svdDownKey) }

            if module is Linear {
                let biasKey = "\(path).bias"
                let linearBias = mappedWeights[biasKey]

                let qp = resolveQuantizationParams(
                    for: module,
                    quantizedWeight: qWeight,
                    scales: scales,
                    fallbackGroupSize: groupSize,
                    fallbackBits: bits
                )
                let resolved = quantizedModuleResolver?(
                    path,
                    module,
                    qWeight,
                    scales,
                    qBiases,
                    qp.groupSize,
                    qp.bits
                ) ?? (groupSize: qp.groupSize, bits: qp.bits, mode: QuantizationMode.affine)

                let quantized: Module
                if useResiduals, let svdUp, let svdDown {
                    quantized = ResidualQuantizedLinear(
                        weight: qWeight,
                        bias: linearBias,
                        scales: scales,
                        biases: qBiases,
                        groupSize: resolved.groupSize,
                        bits: resolved.bits,
                        mode: resolved.mode,
                        residualDown: svdDown,
                        residualUp: svdUp
                    )
                    residualApplied += 1
                } else {
                    quantized = PortableQuantizedLinear(
                        weight: qWeight,
                        bias: linearBias,
                        scales: scales,
                        biases: qBiases,
                        groupSize: resolved.groupSize,
                        bits: resolved.bits,
                        mode: resolved.mode
                    )
                }
                quantizedReplacements[path] = quantized

                if linearBias != nil { usedKeys.insert(biasKey) }
            } else if module is Embedding {
                let qp = resolveQuantizationParams(
                    for: module,
                    quantizedWeight: qWeight,
                    scales: scales,
                    fallbackGroupSize: groupSize,
                    fallbackBits: bits
                )
                let resolved = quantizedModuleResolver?(
                    path,
                    module,
                    qWeight,
                    scales,
                    qBiases,
                    qp.groupSize,
                    qp.bits
                ) ?? (groupSize: qp.groupSize, bits: qp.bits, mode: QuantizationMode.affine)
                let quantized = PreQuantizedEmbedding(
                    weight: qWeight,
                    scales: scales,
                    biases: qBiases,
                    groupSize: resolved.groupSize,
                    bits: resolved.bits,
                    mode: resolved.mode
                )
                quantizedReplacements[path] = quantized
            } else {
                continue
            }

            usedKeys.insert(weightKey)
            usedKeys.insert(scalesKey)
            if qBiases != nil { usedKeys.insert(biasesKey) }
        }

        // Apply module replacements
        if !quantizedReplacements.isEmpty {
            var arrayUpdates: [String: [(Int, Module)]] = [:]
            var directUpdates: [(String, Module)] = []

            for (path, replacement) in quantizedReplacements {
                let components = path.split(separator: ".")
                if let lastComponent = components.last, let index = Int(lastComponent) {
                    let parentPath = components.dropLast().joined(separator: ".")
                    arrayUpdates[parentPath, default: []].append((index, replacement))
                } else {
                    directUpdates.append((path, replacement))
                }
            }

            var moduleUpdates: [(String, Module)] = directUpdates
            for (parentPath, indexedReplacements) in arrayUpdates {
                let currentModules = leafModules.filter { (p, _) in
                    let pComponents = p.split(separator: ".")
                    if pComponents.count < 2 { return false }
                    let pParent = pComponents.dropLast().joined(separator: ".")
                    return pParent == parentPath && Int(pComponents.last!) != nil
                }

                var replacementMap: [Int: Module] = [:]
                for (index, replacement) in indexedReplacements {
                    replacementMap[index] = replacement
                }

                for (modulePath, originalModule) in currentModules {
                    let index = Int(modulePath.split(separator: ".").last!)!
                    let module = replacementMap[index] ?? originalModule
                    moduleUpdates.append((modulePath, module))
                }
            }

            if !moduleUpdates.isEmpty {
                model.update(modules: ModuleChildren.unflattened(moduleUpdates))
            }
        }

        // Apply remaining parameters
        var remainingUpdates: [(String, MLXArray)] = []
        for (key, value) in mappedWeights where !usedKeys.contains(key) {
            remainingUpdates.append(contentsOf: mapper(key, value))
        }

        if !remainingUpdates.isEmpty {
            let verify: Module.VerifyUpdate = Self.verifyUnusedKeys ? .noUnusedKeys : .none
            try model.update(parameters: ModuleParameters.unflattened(remainingUpdates), verify: verify)
        }

        if residualApplied > 0, Self.residualDebug {
            let message = "[HFSafetensorsWeightsLoader] Applied SVD residuals to \(residualApplied) layers\n"
            FileHandle.standardError.write(Data(message.utf8))
        }
    }
}
