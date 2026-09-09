import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func ltxLeakyRelu(_ x: MLXArray, slope: Float) -> MLXArray {
    MLX.maximum(x, x * MLXArray(slope).asType(x.dtype))
}

enum LTXVocoderFlavor: Equatable {
    case legacy
    case bandwidthExtension
}

func detectLTXVocoderFlavor<S: Sequence>(keys: S) -> LTXVocoderFlavor where S.Element == String {
    for key in keys {
        if key.hasPrefix("vocoder.bwe_generator.")
            || key.hasPrefix("vocoder.mel_stft.")
            || key.hasPrefix("vocoder.vocoder.") {
            return .bandwidthExtension
        }
    }
    return .legacy
}

enum LTXVocoderWeightLayout: Equatable {
    case pytorch
    case mlx
}

enum LTXVocoderResBlockKind: Equatable {
    case legacy
    case amp
}

enum LTXVocoderActivationKind: Equatable {
    case leaky
    case snake
    case snakeBeta
}

struct LTXVocoderArchitectureConfig: Equatable {
    let inputChannels: Int
    let outputChannels: Int
    let upsampleInitialChannels: Int
    let upsampleRates: [Int]
    let upsampleKernelSizes: [Int]
    let resblockKernelSizes: [Int]
    let resblockDilationSizes: [[Int]]
    let blockKind: LTXVocoderResBlockKind
    let activation: LTXVocoderActivationKind
    let applyFinalActivation: Bool
    let useTanhAtFinal: Bool
    let useBiasAtFinal: Bool

    static let legacy = LTXVocoderArchitectureConfig(
        inputChannels: 128,
        outputChannels: 2,
        upsampleInitialChannels: 1024,
        upsampleRates: [6, 5, 2, 2, 2],
        upsampleKernelSizes: [16, 15, 8, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        blockKind: .legacy,
        activation: .leaky,
        applyFinalActivation: true,
        useTanhAtFinal: true,
        useBiasAtFinal: true
    )

    static let defaultBWEBase = LTXVocoderArchitectureConfig(
        inputChannels: 128,
        outputChannels: 2,
        upsampleInitialChannels: 1024,
        upsampleRates: [6, 5, 2, 2, 2],
        upsampleKernelSizes: [16, 15, 8, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        blockKind: .amp,
        activation: .snakeBeta,
        applyFinalActivation: true,
        useTanhAtFinal: true,
        useBiasAtFinal: true
    )

    static let defaultBWEGenerator = LTXVocoderArchitectureConfig(
        inputChannels: 128,
        outputChannels: 2,
        upsampleInitialChannels: 1024,
        upsampleRates: [6, 5, 2, 2, 2],
        upsampleKernelSizes: [16, 15, 8, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        blockKind: .amp,
        activation: .snakeBeta,
        applyFinalActivation: false,
        useTanhAtFinal: true,
        useBiasAtFinal: true
    )
}

struct LTXBWEVocoderRuntimeConfig {
    let inputSamplingRate: Int
    let outputSamplingRate: Int
    let hopLength: Int
    let filterLength: Int
    let melChannels: Int
    let baseVocoder: LTXVocoderArchitectureConfig
    let bandwidthExtensionVocoder: LTXVocoderArchitectureConfig
}

struct LTXVocoderModelConfig: Decodable {
    let upsampleInitialChannels: Int?
    let resblock: String?
    let upsampleRates: [Int]?
    let resblockKernelSizes: [Int]?
    let upsampleKernelSizes: [Int]?
    let resblockDilationSizes: [[Int]]?
    let useTanhAtFinal: Bool?
    let activation: String?
    let useBiasAtFinal: Bool?
    let applyFinalActivation: Bool?

    private enum CodingKeys: String, CodingKey {
        case upsampleInitialChannels = "upsample_initial_channel"
        case resblock
        case upsampleRates = "upsample_rates"
        case resblockKernelSizes = "resblock_kernel_sizes"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case useTanhAtFinal = "use_tanh_at_final"
        case activation
        case useBiasAtFinal = "use_bias_at_final"
        case applyFinalActivation = "apply_final_activation"
    }

    func runtimeArchitecture(
        defaultArchitecture: LTXVocoderArchitectureConfig
    ) -> LTXVocoderArchitectureConfig {
        let blockKind: LTXVocoderResBlockKind = switch resblock?.lowercased() {
        case "amp1", "amp":
            .amp
        default:
            defaultArchitecture.blockKind
        }
        let activationKind: LTXVocoderActivationKind = switch activation?.lowercased() {
        case "snake":
            .snake
        case "snakebeta", "snake_beta":
            .snakeBeta
        default:
            defaultArchitecture.activation
        }
        return LTXVocoderArchitectureConfig(
            inputChannels: defaultArchitecture.inputChannels,
            outputChannels: defaultArchitecture.outputChannels,
            upsampleInitialChannels: upsampleInitialChannels ?? defaultArchitecture.upsampleInitialChannels,
            upsampleRates: upsampleRates ?? defaultArchitecture.upsampleRates,
            upsampleKernelSizes: upsampleKernelSizes ?? defaultArchitecture.upsampleKernelSizes,
            resblockKernelSizes: resblockKernelSizes ?? defaultArchitecture.resblockKernelSizes,
            resblockDilationSizes: resblockDilationSizes ?? defaultArchitecture.resblockDilationSizes,
            blockKind: blockKind,
            activation: activationKind,
            applyFinalActivation: applyFinalActivation ?? defaultArchitecture.applyFinalActivation,
            useTanhAtFinal: useTanhAtFinal ?? defaultArchitecture.useTanhAtFinal,
            useBiasAtFinal: useBiasAtFinal ?? defaultArchitecture.useBiasAtFinal
        )
    }

    var hasArchitectureFields: Bool {
        upsampleInitialChannels != nil
            || resblock != nil
            || upsampleRates != nil
            || resblockKernelSizes != nil
            || upsampleKernelSizes != nil
            || resblockDilationSizes != nil
            || useTanhAtFinal != nil
            || activation != nil
            || useBiasAtFinal != nil
            || applyFinalActivation != nil
    }
}

struct LTXBWEVocoderConfig: Decodable {
    let inputSamplingRate: Int
    let outputSamplingRate: Int
    let hopLength: Int
    let filterLength: Int
    let winLength: Int?
    let melChannels: Int
    let modelConfig: LTXVocoderModelConfig

    private enum CodingKeys: String, CodingKey {
        case inputSamplingRate = "input_sampling_rate"
        case outputSamplingRate = "output_sampling_rate"
        case hopLength = "hop_length"
        case filterLength = "n_fft"
        case winLength = "win_size"
        case melChannels = "num_mels"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        inputSamplingRate = try container.decode(Int.self, forKey: .inputSamplingRate)
        outputSamplingRate = try container.decode(Int.self, forKey: .outputSamplingRate)
        hopLength = try container.decode(Int.self, forKey: .hopLength)
        filterLength = try container.decode(Int.self, forKey: .filterLength)
        winLength = try container.decodeIfPresent(Int.self, forKey: .winLength)
        melChannels = try container.decode(Int.self, forKey: .melChannels)
        modelConfig = try LTXVocoderModelConfig(from: decoder)
    }

    func runtimeConfig(baseVocoder: LTXVocoderArchitectureConfig?) -> LTXBWEVocoderRuntimeConfig {
        LTXBWEVocoderRuntimeConfig(
            inputSamplingRate: inputSamplingRate,
            outputSamplingRate: outputSamplingRate,
            hopLength: hopLength,
            filterLength: winLength ?? filterLength,
            melChannels: melChannels,
            baseVocoder: baseVocoder ?? .defaultBWEBase,
            bandwidthExtensionVocoder: modelConfig.runtimeArchitecture(defaultArchitecture: .defaultBWEGenerator)
        )
    }
}

struct LTXVocoderConfigEnvelope: Decodable {
    let baseVocoder: LTXVocoderModelConfig?
    let bwe: LTXBWEVocoderConfig?

    private enum CodingKeys: String, CodingKey {
        case vocoder
        case bwe
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let topLevelBWE = try container.decodeIfPresent(LTXBWEVocoderConfig.self, forKey: .bwe)

        var resolvedBase: LTXVocoderModelConfig?
        var resolvedBWE = topLevelBWE

        if let directBase = try? container.decodeIfPresent(LTXVocoderModelConfig.self, forKey: .vocoder),
           directBase.hasArchitectureFields {
            resolvedBase = directBase
        }

        if let nested = try? container.decodeIfPresent(LTXVocoderConfigNode.self, forKey: .vocoder) {
            if let nestedBase = nested.vocoder, nestedBase.hasArchitectureFields {
                resolvedBase = nestedBase
            }
            if resolvedBWE == nil {
                resolvedBWE = nested.bwe
            }
        }

        self.baseVocoder = resolvedBase
        self.bwe = resolvedBWE
    }
}

struct LTXVocoderConfigNode: Decodable {
    let vocoder: LTXVocoderModelConfig?
    let bwe: LTXBWEVocoderConfig?
}

func loadLTXBWEVocoderConfig(modelRoot: URL) throws -> LTXBWEVocoderRuntimeConfig? {
    let candidates = [
        modelRoot.appendingPathComponent("vocoder/config.json", isDirectory: false),
        modelRoot.appendingPathComponent("embedded_config.json", isDirectory: false),
        modelRoot.appendingPathComponent("config.json", isDirectory: false),
        modelRoot.appendingPathComponent("audio_vae/config.json", isDirectory: false),
    ]
    let decoder = JSONDecoder()

    for url in candidates where FileManager.default.fileExists(atPath: url.path) {
        let data = try Data(contentsOf: url)
        if let direct = try? decoder.decode(LTXBWEVocoderConfig.self, from: data) {
            return direct.runtimeConfig(baseVocoder: nil)
        }
        if let envelope = try? decoder.decode(LTXVocoderConfigEnvelope.self, from: data) {
            if let bwe = envelope.bwe {
                let base = envelope.baseVocoder?.runtimeArchitecture(defaultArchitecture: .defaultBWEBase)
                return bwe.runtimeConfig(baseVocoder: base)
            }
        }
    }

    return nil
}

func loadLTXPackedBWEVocoderConfig(weightsURL: URL) throws -> LTXBWEVocoderRuntimeConfig? {
    let metadata = try SafetensorsStreamingLoader.fileMetadata(url: weightsURL)
    guard let rawConfig = metadata["config"],
          let data = rawConfig.data(using: .utf8),
          let envelope = try? JSONDecoder().decode(LTXVocoderConfigEnvelope.self, from: data),
          let bwe = envelope.bwe else {
        return nil
    }
    let base = envelope.baseVocoder?.runtimeArchitecture(defaultArchitecture: .defaultBWEBase)
    return bwe.runtimeConfig(baseVocoder: base)
}
