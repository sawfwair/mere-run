import Foundation
import MLX
import MLXNN

public struct ACEStepResources: Sendable, Hashable {
    public var modelRootURL: URL

    public init(rootURL: URL) {
        self.modelRootURL = rootURL
    }

    public var configURL: URL {
        modelRootURL.appending(path: "config.json")
    }

    public var weightsIndexURL: URL {
        modelRootURL.appending(path: "model.safetensors.index.json")
    }

    public var weightsURL: URL {
        modelRootURL.appending(path: "model.safetensors")
    }

    public var silenceLatentURL: URL {
        modelRootURL.appending(path: "silence_latent.pt")
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var urls: [URL] = [configURL]

        let weightsOK =
            fileManager.fileExists(atPath: weightsIndexURL.path)
            || fileManager.fileExists(atPath: weightsURL.path)
        if !weightsOK {
            urls.append(weightsIndexURL)
        }

        return urls.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}

enum ACEStepCheckpointLoader {
    enum LoaderError: LocalizedError {
        case missingFiles([URL])
        case invalidSilenceLatent(String)

        var errorDescription: String? {
            switch self {
            case .missingFiles(let urls):
                let list = urls.map { $0.path }.joined(separator: "\n")
                return "Missing ACE-Step resources:\n\(list)"
            case .invalidSilenceLatent(let reason):
                return "Invalid ACE-Step silence latent: \(reason)"
            }
        }
    }

    static func loadConfig(
        resources: ACEStepResources,
        fileManager: FileManager = .default
    ) throws -> ACEStepConfig {
        let missing = resources.validate(fileManager: fileManager)
        if !missing.isEmpty {
            throw LoaderError.missingFiles(missing)
        }
        let data = try Data(contentsOf: resources.configURL)
        return try JSONDecoder().decode(ACEStepConfig.self, from: data)
    }

    static func loadDecoder(
        resources: ACEStepResources,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .noUnusedKeys,
        fileManager: FileManager = .default
    ) throws -> ACEStepDiT {
        let config = try loadConfig(resources: resources, fileManager: fileManager)
        let model = ACEStepDecoderOnlyModel(config: config)

        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.weightsIndexURL,
            singleURL: resources.weightsURL,
            to: model,
            dtype: dtype,
            verify: verify,
            mapper: { key, value in
                guard key.hasPrefix("decoder.") else { return [] }

                if key == "decoder.proj_in.1.weight", value.ndim == 3 {
                    let t = value.transposed(0, 2, 1)
                    return [("decoder.proj_in.weight", t.reshaped(-1).reshaped(t.shape))]
                }

                if key == "decoder.proj_in.1.bias" {
                    return [("decoder.proj_in.bias", value)]
                }

                if key == "decoder.proj_out.1.weight", value.ndim == 3 {
                    let t = value.transposed(1, 2, 0)
                    return [("decoder.proj_out.weight", t.reshaped(-1).reshaped(t.shape))]
                }

                if key == "decoder.proj_out.1.bias" {
                    return [("decoder.proj_out.bias", value)]
                }

                return [(key, value)]
            },
            fileManager: fileManager
        )

        return model.decoder
    }

    static func loadDecoderBundle(
        resources: ACEStepResources,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .noUnusedKeys,
        fileManager: FileManager = .default
    ) throws -> (decoder: ACEStepDiT, nullConditionEmbedding: MLXArray) {
        let config = try loadConfig(resources: resources, fileManager: fileManager)
        let model = ACEStepDecoderBundleModel(config: config)

        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.weightsIndexURL,
            singleURL: resources.weightsURL,
            to: model,
            dtype: dtype,
            verify: verify,
            mapper: { key, value in
                if key == "null_condition_emb" {
                    return [(key, value)]
                }

                guard key.hasPrefix("decoder.") else { return [] }

                if key == "decoder.proj_in.1.weight", value.ndim == 3 {
                    let t = value.transposed(0, 2, 1)
                    return [("decoder.proj_in.weight", t.reshaped(-1).reshaped(t.shape))]
                }

                if key == "decoder.proj_in.1.bias" {
                    return [("decoder.proj_in.bias", value)]
                }

                if key == "decoder.proj_out.1.weight", value.ndim == 3 {
                    let t = value.transposed(1, 2, 0)
                    return [("decoder.proj_out.weight", t.reshaped(-1).reshaped(t.shape))]
                }

                if key == "decoder.proj_out.1.bias" {
                    return [("decoder.proj_out.bias", value)]
                }

                return [(key, value)]
            },
            fileManager: fileManager
        )

        return (model.decoder, model.nullConditionEmbedding)
    }

    static func loadSilenceLatentIfPresent(
        resources: ACEStepResources,
        latentDim: Int,
        dtype: DType? = .bfloat16,
        fileManager: FileManager = .default
    ) throws -> MLXArray? {
        guard fileManager.fileExists(atPath: resources.silenceLatentURL.path) else {
            return nil
        }
        return try ACEStepSilenceLatentLoader.load(
            from: resources.silenceLatentURL,
            latentDim: latentDim,
            dtype: dtype
        )
    }

    static func loadTurboBundle(
        resources: ACEStepResources,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .noUnusedKeys,
        fileManager: FileManager = .default
    ) throws -> (
        decoder: ACEStepDiT,
        encoder: ACEStepConditionEncoder,
        tokenizer: ACEStepAudioTokenizer,
        detokenizer: ACEStepAudioTokenDetokenizer,
        nullConditionEmbedding: MLXArray
    ) {
        let config = try loadConfig(resources: resources, fileManager: fileManager)
        let model = ACEStepTurboBundleModel(config: config)

        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.weightsIndexURL,
            singleURL: resources.weightsURL,
            to: model,
            dtype: dtype,
            verify: verify,
            mapper: { key, value in
                if key == "null_condition_emb" {
                    return [(key, value)]
                }

                if key == "decoder.proj_in.1.weight", value.ndim == 3 {
                    let t = value.transposed(0, 2, 1)
                    return [("decoder.proj_in.weight", t.reshaped(-1).reshaped(t.shape))]
                }

                if key == "decoder.proj_in.1.bias" {
                    return [("decoder.proj_in.bias", value)]
                }

                if key == "decoder.proj_out.1.weight", value.ndim == 3 {
                    let t = value.transposed(1, 2, 0)
                    return [("decoder.proj_out.weight", t.reshaped(-1).reshaped(t.shape))]
                }

                if key == "decoder.proj_out.1.bias" {
                    return [("decoder.proj_out.bias", value)]
                }

                return [(key, value)]
            },
            fileManager: fileManager
        )

        return (
            decoder: model.decoder,
            encoder: model.encoder,
            tokenizer: model.tokenizer,
            detokenizer: model.detokenizer,
            nullConditionEmbedding: model.nullConditionEmbedding
        )
    }
}

private enum ACEStepSilenceLatentLoader {
    private struct ZipEntry {
        var name: String
        var compressionMethod: UInt16
        var compressedSize: Int
        var localHeaderOffset: Int
    }

    static func load(from url: URL, latentDim: Int, dtype: DType?) throws -> MLXArray {
        guard latentDim > 0 else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent("latentDim must be positive.")
        }
        let archive = try Data(contentsOf: url)
        let tensorData = try storedEntryData(namedSuffix: "/data/0", in: archive)

        if let byteOrder = try? storedEntryData(namedSuffix: "/byteorder", in: archive) {
            let value = String(decoding: byteOrder, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard value == "little" else {
                throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                    "unsupported byte order '\(value)' in \(url.lastPathComponent)."
                )
            }
        }

        guard tensorData.count % MemoryLayout<Float>.size == 0 else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                "tensor byte count \(tensorData.count) is not divisible by float32 size."
            )
        }

        let valueCount = tensorData.count / MemoryLayout<Float>.size
        guard valueCount % latentDim == 0 else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                "float count \(valueCount) is not divisible by latent dim \(latentDim)."
            )
        }

        var values: [Float] = []
        values.reserveCapacity(valueCount)
        for index in 0..<valueCount {
            let bits = try tensorData.readUInt32LE(at: index * MemoryLayout<Float>.size)
            values.append(Float(bitPattern: bits))
        }

        let frames = valueCount / latentDim
        let channelFirst = MLXArray(values, [1, latentDim, frames])
        return channelFirst.transposed(0, 2, 1).asType(dtype ?? .float32)
    }

    private static func storedEntryData(namedSuffix suffix: String, in archive: Data) throws -> Data {
        let entries = try readCentralDirectory(from: archive)
        guard let entry = entries.first(where: { $0.name.hasSuffix(suffix) }) else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                "missing zip entry matching \(suffix)."
            )
        }
        guard entry.compressionMethod == 0 else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                "\(entry.name) uses unsupported compression method \(entry.compressionMethod)."
            )
        }

        let localOffset = entry.localHeaderOffset
        guard try archive.readUInt32LE(at: localOffset) == 0x0403_4b50 else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                "\(entry.name) local header has an invalid signature."
            )
        }
        let nameLength = Int(try archive.readUInt16LE(at: localOffset + 26))
        let extraLength = Int(try archive.readUInt16LE(at: localOffset + 28))
        let dataStart = localOffset + 30 + nameLength + extraLength
        let dataEnd = dataStart + entry.compressedSize
        guard dataStart >= 0, dataEnd <= archive.count else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                "\(entry.name) data range is outside the archive."
            )
        }
        return archive.subdata(in: dataStart..<dataEnd)
    }

    private static func readCentralDirectory(from archive: Data) throws -> [ZipEntry] {
        let eocdOffset = try findEndOfCentralDirectory(in: archive)
        let entryCount = Int(try archive.readUInt16LE(at: eocdOffset + 10))
        let centralDirectoryOffset = Int(try archive.readUInt32LE(at: eocdOffset + 16))

        var entries: [ZipEntry] = []
        entries.reserveCapacity(entryCount)
        var offset = centralDirectoryOffset
        for _ in 0..<entryCount {
            guard try archive.readUInt32LE(at: offset) == 0x0201_4b50 else {
                throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                    "central directory has an invalid signature."
                )
            }

            let compressionMethod = try archive.readUInt16LE(at: offset + 10)
            let compressedSize = Int(try archive.readUInt32LE(at: offset + 20))
            let nameLength = Int(try archive.readUInt16LE(at: offset + 28))
            let extraLength = Int(try archive.readUInt16LE(at: offset + 30))
            let commentLength = Int(try archive.readUInt16LE(at: offset + 32))
            let localHeaderOffset = Int(try archive.readUInt32LE(at: offset + 42))
            let nameStart = offset + 46
            let nameEnd = nameStart + nameLength
            guard nameEnd <= archive.count else {
                throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
                    "central directory filename is outside the archive."
                )
            }

            let name = String(decoding: archive[nameStart..<nameEnd], as: UTF8.self)
            entries.append(
                ZipEntry(
                    name: name,
                    compressionMethod: compressionMethod,
                    compressedSize: compressedSize,
                    localHeaderOffset: localHeaderOffset
                )
            )
            offset = nameEnd + extraLength + commentLength
        }

        return entries
    }

    private static func findEndOfCentralDirectory(in archive: Data) throws -> Int {
        guard archive.count >= 22 else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent("zip archive is too small.")
        }

        let minOffset = max(0, archive.count - 22 - 65_535)
        var offset = archive.count - 22
        while offset >= minOffset {
            if try archive.readUInt32LE(at: offset) == 0x0605_4b50 {
                return offset
            }
            offset -= 1
        }

        throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent(
            "missing zip end-of-central-directory record."
        )
    }
}

private extension Data {
    func readUInt16LE(at offset: Int) throws -> UInt16 {
        guard offset >= 0, offset + 2 <= count else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent("unexpected end of data.")
        }
        return UInt16(self[offset])
            | (UInt16(self[offset + 1]) << 8)
    }

    func readUInt32LE(at offset: Int) throws -> UInt32 {
        guard offset >= 0, offset + 4 <= count else {
            throw ACEStepCheckpointLoader.LoaderError.invalidSilenceLatent("unexpected end of data.")
        }
        return UInt32(self[offset])
            | (UInt32(self[offset + 1]) << 8)
            | (UInt32(self[offset + 2]) << 16)
            | (UInt32(self[offset + 3]) << 24)
    }
}

final class ACEStepDecoderOnlyModel: Module {
    @ModuleInfo(key: "decoder") var decoder: ACEStepDiT

    init(config: ACEStepConfig) {
        self._decoder.wrappedValue = ACEStepDiT(config: config)
    }
}

final class ACEStepDecoderBundleModel: Module {
    @ModuleInfo(key: "decoder") var decoder: ACEStepDiT
    @ParameterInfo(key: "null_condition_emb") var nullConditionEmbedding: MLXArray

    init(config: ACEStepConfig) {
        self._decoder.wrappedValue = ACEStepDiT(config: config)
        self._nullConditionEmbedding.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
    }
}

final class ACEStepTurboBundleModel: Module {
    @ModuleInfo(key: "decoder") var decoder: ACEStepDiT
    @ModuleInfo(key: "encoder") var encoder: ACEStepConditionEncoder
    @ModuleInfo(key: "tokenizer") var tokenizer: ACEStepAudioTokenizer
    @ModuleInfo(key: "detokenizer") var detokenizer: ACEStepAudioTokenDetokenizer
    @ParameterInfo(key: "null_condition_emb") var nullConditionEmbedding: MLXArray

    init(config: ACEStepConfig) {
        self._decoder.wrappedValue = ACEStepDiT(config: config)
        self._encoder.wrappedValue = ACEStepConditionEncoder(config: config)
        self._tokenizer.wrappedValue = ACEStepAudioTokenizer(config: config)
        self._detokenizer.wrappedValue = ACEStepAudioTokenDetokenizer(config: config)
        self._nullConditionEmbedding.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
    }
}
