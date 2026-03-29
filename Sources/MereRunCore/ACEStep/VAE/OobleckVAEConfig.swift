import Foundation
import MLX
import MLXNN

public struct OobleckVAEConfig: Decodable, Sendable, Hashable {
    public let audioChannels: Int
    public let channelMultiples: [Int]
    public let decoderChannels: Int
    public let decoderInputChannels: Int
    public let downsamplingRatios: [Int]
    public let encoderHiddenSize: Int
    public let samplingRate: Int

    enum CodingKeys: String, CodingKey {
        case audioChannels = "audio_channels"
        case channelMultiples = "channel_multiples"
        case decoderChannels = "decoder_channels"
        case decoderInputChannels = "decoder_input_channels"
        case downsamplingRatios = "downsampling_ratios"
        case encoderHiddenSize = "encoder_hidden_size"
        case samplingRate = "sampling_rate"
    }

    public init(
        audioChannels: Int = 2,
        channelMultiples: [Int] = [1, 2, 4, 8, 16],
        decoderChannels: Int = 128,
        decoderInputChannels: Int = 64,
        downsamplingRatios: [Int] = [2, 4, 4, 6, 10],
        encoderHiddenSize: Int = 128,
        samplingRate: Int = 48_000
    ) {
        self.audioChannels = audioChannels
        self.channelMultiples = channelMultiples
        self.decoderChannels = decoderChannels
        self.decoderInputChannels = decoderInputChannels
        self.downsamplingRatios = downsamplingRatios
        self.encoderHiddenSize = encoderHiddenSize
        self.samplingRate = samplingRate
    }
}

public struct OobleckVAEResources: Sendable, Hashable {
    public var modelRootURL: URL

    public init(rootURL: URL) {
        self.modelRootURL = rootURL
    }

    public var configURL: URL {
        modelRootURL.appending(path: "config.json")
    }

    public var weightsIndexURL: URL {
        modelRootURL.appending(path: "diffusion_pytorch_model.safetensors.index.json")
    }

    public var weightsURL: URL {
        modelRootURL.appending(path: "diffusion_pytorch_model.safetensors")
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

enum OobleckVAECheckpointLoader {
    enum LoaderError: LocalizedError {
        case missingFiles([URL])

        var errorDescription: String? {
            switch self {
            case .missingFiles(let urls):
                let list = urls.map { $0.path }.joined(separator: "\n")
                return "Missing Oobleck VAE resources:\n\(list)"
            }
        }
    }

    static func loadConfig(
        resources: OobleckVAEResources,
        fileManager: FileManager = .default
    ) throws -> OobleckVAEConfig {
        let missing = resources.validate(fileManager: fileManager)
        if !missing.isEmpty {
            throw LoaderError.missingFiles(missing)
        }
        let data = try Data(contentsOf: resources.configURL)
        return try JSONDecoder().decode(OobleckVAEConfig.self, from: data)
    }

    static func loadVAE(
        resources: OobleckVAEResources,
        dtype: DType? = .bfloat16,
        verify: Module.VerifyUpdate = .noUnusedKeys,
        fileManager: FileManager = .default
    ) throws -> OobleckVAE {
        let config = try loadConfig(resources: resources, fileManager: fileManager)
        let model = OobleckVAE(config: config)

        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.weightsIndexURL,
            singleURL: resources.weightsURL,
            to: model,
            dtype: dtype,
            verify: verify,
            mapper: { key, value in
                guard key.hasPrefix("decoder.") || key.hasPrefix("encoder.") else { return [] }

                if key.hasSuffix(".weight_v"), value.ndim == 3 {
                    let t: MLXArray
                    if key.contains(".conv_t") {
                        // PyTorch ConvTranspose1d: [in, out, k] -> MLX: [out, k, in]
                        t = value.transposed(1, 2, 0)
                    } else {
                        // PyTorch Conv1d: [out, in, k] -> MLX: [out, k, in]
                        t = value.transposed(0, 2, 1)
                    }
                    return [(key, t.reshaped(-1).reshaped(t.shape))]
                }

                if (key.hasSuffix(".alpha") || key.hasSuffix(".beta")), key.contains(".snake"), value.ndim == 3 {
                    // PyTorch Snake params: [1, C, 1] -> NLC broadcast: [1, 1, C]
                    let t = value.transposed(0, 2, 1)
                    return [(key, t.reshaped(-1).reshaped(t.shape))]
                }

                return [(key, value)]
            },
            fileManager: fileManager
        )

        return model
    }
}
