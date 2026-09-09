import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXUnifiedAVGenerator {
    @discardableResult
    public func loadTextToAudio(
        modelRoot: URL,
        dtype: DType = .bfloat16
    ) async throws -> LTXLoadTimings {
        let totalStart = ltxMonotonicSeconds()
        await unload()
        let root = modelRoot.standardizedFileURL
        guard isLTX25FullModelRoot(root) else {
            throw LTXUnifiedAVGeneratorError.textToAudioRequiresLTX25Full(root)
        }
        let resources = LTX25Resources(rootURL: root)
        let transformerURL = resolvedLTX25TransformerURL(resources: resources, kind: .dev)
        let audioWeightsURL = resources.audioVAEURL
        guard FileManager.default.fileExists(atPath: transformerURL.path) else {
            throw LTXUnifiedAVGeneratorError.transformerWeightsMissing(transformerURL)
        }
        guard FileManager.default.fileExists(atPath: audioWeightsURL.path) else {
            throw LTXUnifiedAVGeneratorError.audioVAEWeightsMissing(audioWeightsURL)
        }

        let textStart = ltxMonotonicSeconds()
        let text = LTXGemmaTextEncoder()
        try await text.load(
            modelRoot: root,
            textEncoderRoot: nil,
            dtype: dtype,
            loadConnectorWeights: true
        )
        let textSeconds = ltxMonotonicSeconds() - textStart

        let transformerStart = ltxMonotonicSeconds()
        let model = LTXAudioOnlyTransformerV2()
        try loadLTX25TransformerWeights(
            url: transformerURL,
            model: model,
            dtype: dtype,
            sourceInclude: isLTXAudioOnlyTransformerWeight,
            nativeInclude: {
                isLTXAudioOnlyTransformerWeight("model.diffusion_model.\($0)")
            }
        )
        let transformerSeconds = ltxMonotonicSeconds() - transformerStart

        let decoderStart = ltxMonotonicSeconds()
        let activeAudioDecoder = try loadLTXAudioDecoder(
            weightsURL: audioWeightsURL,
            sourceLayout: .pytorch
        )
        let activeVocoder = try loadLTXVocoder(
            weightsURL: audioWeightsURL,
            sourceLayout: .pytorch,
            configurationRoot: root,
            usesPackedConfiguration: true
        )
        let decoderSeconds = ltxMonotonicSeconds() - decoderStart

        textEncoder = text
        audioOnlyTransformer = model
        audioDecoder = activeAudioDecoder
        vocoder = activeVocoder
        loadedRoot = root
        loadedDType = dtype
        loadedForLTX25 = true
        return LTXLoadTimings(
            textEncoderSeconds: textSeconds,
            transformerSeconds: transformerSeconds,
            audioDecoderSeconds: decoderSeconds,
            totalSeconds: ltxMonotonicSeconds() - totalStart
        )
    }
}
