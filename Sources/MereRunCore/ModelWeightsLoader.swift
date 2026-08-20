import Foundation
import MLX
import MLXNN

/// Shared weight-loading utilities used by both image model families.
///
/// Goals:
/// - One canonical decision tree for: sharded-index vs single-file, quantized vs full precision.
/// - Centralize dtype casting + verify options.
/// - Allow model-family call sites to provide only mapping hooks.
public enum ModelWeightsLoader {
    public struct QuantizationParams: Hashable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public let svdResidualRank: Int?

        public init(bits: Int, groupSize: Int, svdResidualRank: Int? = nil) {
            self.bits = bits
            self.groupSize = groupSize
            self.svdResidualRank = svdResidualRank
        }

        public var applySVDResiduals: Bool? {
            svdResidualRank.map { $0 > 0 }
        }

        public static func fromManifest(_ manifest: MereRunModelManifest?) throws -> QuantizationParams? {
            guard let manifest else { return nil }

            let precisionImpliesQuant = manifest.precision == .int1
                || manifest.precision == .int2
                || manifest.precision == .int4
                || manifest.precision == .int6
                || manifest.precision == .int8
            let hasQuant = manifest.quantization != nil
            guard precisionImpliesQuant || hasQuant else {
                return nil
            }

            guard let q = manifest.quantization else {
                throw LoaderError.invalidQuantizationMetadata("Missing quantization metadata for precision=\(manifest.precision?.rawValue ?? "unknown")")
            }
            guard let bits = q.bits else {
                throw LoaderError.invalidQuantizationMetadata("Missing quantization.bits")
            }
            guard let groupSize = q.groupSize else {
                throw LoaderError.invalidQuantizationMetadata("Missing quantization.groupSize")
            }
            let svdResidualRank: Int?
            if let rank = q.svdResidualRank {
                guard rank >= 0 else {
                    throw LoaderError.invalidQuantizationMetadata("Invalid quantization.svdResidualRank=\(rank) (expected >= 0)")
                }
                svdResidualRank = rank
            } else {
                svdResidualRank = nil
            }
            return QuantizationParams(bits: bits, groupSize: groupSize, svdResidualRank: svdResidualRank)
        }
    }

    public enum LoaderError: LocalizedError, Sendable {
        case weightsNotFound(indexURL: URL, singleURL: URL)
        case noSafetensorsShards(URL)
        case invalidQuantizationMetadata(String)

        public var errorDescription: String? {
            switch self {
            case .weightsNotFound(let indexURL, let singleURL):
                return "No weights found at \(indexURL.path) or \(singleURL.path)"
            case .noSafetensorsShards(let url):
                return "No safetensors shard files found in \(url.path)"
            case .invalidQuantizationMetadata(let message):
                return "Invalid quantization metadata: \(message)"
            }
        }
    }

    // MARK: - Hugging Face safetensors (index or single)

    /// Applies weights from either a sharded index (`*.safetensors.index.json`) or a single safetensors file.
    ///
    /// - If `quantization` is provided, the loader will prefer the quantized path.
    /// - Otherwise, quantization is detected:
    ///   - from the index file's weight map (if index exists), or
    ///   - from `.scales` keys in the single safetensors file.
    public static func applyHFSafetensors(
        indexURL: URL,
        singleURL: URL,
        to model: Module,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .none,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] },
        keyMapper: ((String) -> String)? = nil,
        quantization: QuantizationParams? = nil,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil,
        fileManager: FileManager = .default
    ) throws {
        let hasIndex = fileManager.fileExists(atPath: indexURL.path)
        let hasSingle = fileManager.fileExists(atPath: singleURL.path)

        if hasIndex {
            let isQuantized = isQuantizedIndex(indexURL: indexURL, fileManager: fileManager)
            if isQuantized {
                guard let q = quantization else {
                    throw LoaderError.invalidQuantizationMetadata(
                        "Quantized weights detected in index, but no quantization params were provided (missing \(MereRunModelManifest.filename))."
                    )
                }
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: indexURL,
                    to: model,
                    groupSize: q.groupSize,
                    bits: q.bits,
                    applySVDResiduals: q.applySVDResiduals,
                    keyMapper: keyMapper,
                    mapper: mapper,
                    progressHandler: progressHandler
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: indexURL,
                    to: model,
                    dtype: dtype,
                    verify: verify,
                    mapper: mapper,
                    progressHandler: progressHandler
                )
            }
            return
        }

        if hasSingle {
            let arrays = try MLX.loadArrays(url: singleURL)
            let isQuantized = HFSafetensorsWeightsLoader.isQuantized(arrays)
            if isQuantized {
                guard let q = quantization else {
                    throw LoaderError.invalidQuantizationMetadata(
                        "Quantized weights detected in safetensors, but no quantization params were provided (missing \(MereRunModelManifest.filename))."
                    )
                }
                try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                    arrays,
                    to: model,
                    groupSize: q.groupSize,
                    bits: q.bits,
                    applySVDResiduals: q.applySVDResiduals,
                    keyMapper: keyMapper,
                    mapper: mapper
                )
            } else {
                try applyNonQuantizedWeightsFromArrays(
                    arrays,
                    to: model,
                    dtype: dtype,
                    verify: verify,
                    mapper: mapper
                )
            }
            return
        }

        throw LoaderError.weightsNotFound(indexURL: indexURL, singleURL: singleURL)
    }

    private static func isQuantizedIndex(indexURL: URL, fileManager: FileManager) -> Bool {
        guard fileManager.fileExists(atPath: indexURL.path),
              let data = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) else {
            return false
        }
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }

    private static func applyNonQuantizedWeightsFromArrays(
        _ arrays: [String: MLXArray],
        to model: Module,
        dtype: DType?,
        verify: Module.VerifyUpdate,
        mapper: (String, MLXArray) -> [(String, MLXArray)]
    ) throws {
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(arrays.count)
        for (key, rawValue) in arrays {
            let value = HFSafetensorsWeightsLoader.castIfNeeded(rawValue, dtype: dtype)
            updates.append(contentsOf: mapper(key, value))
        }

        if updates.isEmpty {
            return
        }

        try model.update(parameters: ModuleParameters.unflattened(updates), verify: verify)
    }

    // MARK: - mflux-style shards (0.safetensors, 1.safetensors, ...)

    public static func safetensorsShards(in directoryURL: URL, fileManager: FileManager = .default) throws -> [URL] {
        let urls = try fileManager.contentsOfDirectoryResolvingSymlinks(
            at: directoryURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return urls
    }

    /// Applies weights from a list of safetensors shard files.
    ///
    /// This is used for "mflux-exported" directories that don't ship a `*.index.json`.
    public static func applySafetensorsShards(
        files: [URL],
        to model: Module,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .none,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] },
        keyMapper: ((String) -> String)? = nil,
        quantization: QuantizationParams? = nil
    ) throws {
        guard let first = files.first else {
            throw LoaderError.noSafetensorsShards(URL(fileURLWithPath: "."))
        }

        let firstArrays = try MLX.loadArrays(url: first)
        let isQuantized = HFSafetensorsWeightsLoader.isQuantized(firstArrays)

        if isQuantized {
            guard let q = quantization else {
                throw LoaderError.invalidQuantizationMetadata(
                    "Quantized weights detected in safetensors shards, but no quantization params were provided (missing \(MereRunModelManifest.filename))."
                )
            }
            var allWeights: [String: MLXArray] = [:]
            allWeights.reserveCapacity(firstArrays.count * max(1, files.count))
            for (k, v) in firstArrays {
                allWeights[k] = v
            }
            for file in files.dropFirst() {
                let arrays = try MLX.loadArrays(url: file)
                for (k, v) in arrays {
                    allWeights[k] = v
                }
            }
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                allWeights,
                to: model,
                groupSize: q.groupSize,
                bits: q.bits,
                applySVDResiduals: q.applySVDResiduals,
                keyMapper: keyMapper,
                mapper: mapper
            )
        } else {
            // Apply each shard sequentially.
            try applyNonQuantizedWeightsFromArrays(
                firstArrays,
                to: model,
                dtype: dtype,
                verify: verify,
                mapper: mapper
            )
            for file in files.dropFirst() {
                let arrays = try MLX.loadArrays(url: file)
                try applyNonQuantizedWeightsFromArrays(
                    arrays,
                    to: model,
                    dtype: dtype,
                    verify: verify,
                    mapper: mapper
                )
            }
        }
    }
}
