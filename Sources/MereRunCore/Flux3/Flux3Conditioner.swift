import Foundation
import MLX
import MLXNN

final class Flux3Conditioner {
    let encoder: QwenTextEncoder
    let tokenizer: QwenTokenizer

    init(resources: Flux3Resources) throws {
        let root = resources.rootURL.appending(path: "text_encoder")
        let config = try JSONDecoder().decode(Qwen3VLEmbeddingRootConfig.self,
            from: Data(contentsOf: root.appending(path: "config.json"))).textConfig
        guard config.hiddenSize == 2560, config.numHiddenLayers >= 32 else {
            throw Flux3Error.invalidConfiguration("Expected the Qwen3-VL-4B text encoder.")
        }
        encoder = QwenTextEncoder(configuration: QwenTextEncoderConfiguration(
            vocabSize: config.vocabSize, hiddenSize: config.hiddenSize, numHiddenLayers: config.numHiddenLayers,
            numAttentionHeads: config.numAttentionHeads, numKeyValueHeads: config.numKeyValueHeads ?? config.numAttentionHeads,
            intermediateSize: config.intermediateSize, ropeTheta: config.ropeTheta ?? 5_000_000,
            maxPositionEmbeddings: config.maxPositionEmbeddings ?? 262_144, rmsNormEps: config.rmsNormEps ?? 1e-6,
            headDim: config.headDim ?? 128, mropeSection: config.ropeScaling?.mropeSection ?? [24, 20, 20],
            mropeInterleaved: config.ropeScaling?.mropeInterleaved ?? true))
        var mapped: [String: MLXArray] = [:]
        for (key, value) in try resources.textArrays() {
            if key.hasPrefix("model.visual.") || key == "lm_head.weight" { continue }
            guard key.hasPrefix("model.language_model.") else {
                throw Flux3Error.invalidWeights("Unexpected text encoder tensor: \(key).")
            }
            mapped["encoder." + key.dropFirst("model.language_model.".count)] = value.asType(.bfloat16)
        }
        let expected = Dictionary(uniqueKeysWithValues: encoder.parameters().flattened().map { ($0.0, $0.1.shape) })
        guard Set(mapped.keys) == Set(expected.keys), expected.allSatisfy({ mapped[$0.key]?.shape == $0.value }) else {
            throw Flux3Error.invalidWeights("Text encoder names or shapes differ from Qwen3-VL-4B.")
        }
        try encoder.update(parameters: ModuleParameters.unflattened(mapped), verify: [.all])
        tokenizer = try QwenTokenizer.load(from: root, maxLengthOverride: Flux3PromptPlan.maximumLength)
        eval(encoder)
    }

    func encode(_ prompt: String) throws -> MLXArray {
        let real = Array(tokenizer.encodeText(Flux3PromptPlan.presentation(prompt)).prefix(Flux3PromptPlan.maximumLength))
        let length = Flux3PromptPlan.paddedLength(tokenCount: real.count)
        let ids = real + Array(repeating: tokenizer.padTokenId, count: length - real.count)
        let mask = Array(repeating: Int32(1), count: real.count) + Array(repeating: Int32(0), count: length - real.count)
        let positions = (0..<length).map { Int32($0 < real.count ? $0 : 1) }
        let input = MLXArray(ids.map(Int32.init)).reshaped(1, length)
        var hidden = encoder.encoder.forwardActivationHiddenStates(
            embeddings: encoder.encoder.embed(inputIds: input), attentionMask: MLXArray(mask).reshaped(1, length),
            positionIds: MLXArray(positions).reshaped(1, length),
            activationLayers: Flux3PromptPlan.outputLayers.map { $0 - 1 })
        guard hidden.count == Flux3PromptPlan.outputLayers.count else {
            throw Flux3Error.invalidConfiguration("Missing text conditioning layers.")
        }
        // Hugging Face hidden_states includes the final norm only at the final layer.
        if encoder.configuration.numHiddenLayers == Flux3PromptPlan.outputLayers.last {
            hidden[hidden.count - 1] = encoder.encoder.norm(hidden[hidden.count - 1])
        }
        let tokens = concatenated(hidden, axis: -1).reshaped(length, 20480).asType(.bfloat16)
        eval(tokens)
        return tokens
    }
}
