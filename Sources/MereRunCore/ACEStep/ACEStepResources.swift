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

        var errorDescription: String? {
            switch self {
            case .missingFiles(let urls):
                let list = urls.map { $0.path }.joined(separator: "\n")
                return "Missing ACE-Step resources:\n\(list)"
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
