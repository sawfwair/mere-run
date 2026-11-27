import Foundation

enum NemotronImageResources {
    static let repository = "nvidia/Nemotron-Labs-Image-30B-A3B"
    static let revision = "d7c2ba6a59a384080f0bdd5b1e846da03804cdd1"
    static let decoderDirectory = "Nemotron-Labs-Image-Tokenizer"
    static let shards = (1...32).map { String(format: "model-%05d-of-00032.safetensors", $0) }
    static let supportFiles = [
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "unitron_extension_manifest.json",
        "\(decoderDirectory)/config.json", "\(decoderDirectory)/model.safetensors"
    ]
    static let files = supportFiles + ["model.safetensors.index.json"] + shards

    static func read<T: Decodable>(_ type: T.Type, at url: URL) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }

    static func invalid(_ message: String) -> ImageGenerationIssue {
        ImageGenerationIssue("nemotron_image_checkpoint_invalid", message)
    }

    static func languageConfig(at root: URL) throws -> NemotronHConfig {
        let value = try read(NemotronOmniLanguageConfig.self, at: root.appending(path: "config.json"))
        guard value.modelType == "nemotron_h", value.vocabSize == 327_680, value.hiddenSize == 2_688,
              value.hybridOverridePattern == "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
              value.numHiddenLayers == 52, value.nRoutedExperts == 128, value.numExpertsPerToken == 6,
              value.nSharedExperts == 1, value.nGroup == 1, value.topKGroup == 1,
              value.mambaNumHeads == 64, value.mambaHeadDim == 64, value.ssmStateSize == 128,
              value.nGroups == 8, value.convKernel == 4, value.numAttentionHeads == 32,
              value.numKeyValueHeads == 2, value.headDim == 128, value.moeIntermediateSize == 1_856,
              value.sharedExpertIntermediateSize == 3_712 else {
            throw invalid("Expected the released Nemotron-Labs-Image-30B-A3B architecture.")
        }
        // The source model's runtime dt_limit starts at zero. time_step_min is for initialization.
        return value.runtimeConfig(minimumTimeStep: 0)
    }

    struct DecoderConfig: Decodable {
        let modelType: String
        let patchSize: Int
        let width: Int
        let heads: Int
        let layers: Int
        let embedDim: Int
        let nEmbed: Int
        let l2Norm: Bool
        let useFfnOutput: Bool
        let dimFfnOutput: Int
        let mlpRatio: Double
        let vitRopeTheta: Double
        let clampRange: [Double]

        static func validate(at root: URL) throws {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let config = try decoder.decode(Self.self, from: Data(contentsOf: root.appending(path: "config.json")))
            guard config.modelType == "l24sq_vit", config.patchSize == 16, config.width == 1_024,
                  config.heads == 16, config.layers == 24, config.embedDim == 24, config.nEmbed == 196_560,
                  config.l2Norm, config.useFfnOutput, config.dimFfnOutput == 3_072, config.mlpRatio == 4,
                  config.vitRopeTheta == 10_000, config.clampRange == [-1, 1] else {
                throw invalid("Expected the bundled L24SQ visual decoder.")
            }
        }
    }
}
