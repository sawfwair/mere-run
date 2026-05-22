import Foundation
import MLX
import MLXNN
import MLXRandom

extension ZImageTurboLoRATrainer {
    // MARK: - Dataset Encoding

    static func encodePromptEmbeds(
        prompt: String,
        tokenizer: QwenTokenizer,
        textEncoder: QwenTextEncoder
    ) throws -> MLXArray {
        let encoded = try tokenizer.encodeChat(prompts: [prompt], maxLength: tokenizer.maxLength)

        // Use encodeForZImage which is the same method the generator uses
        let embeddingsList = textEncoder.encodeForZImage(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask
        )

        guard let firstEmbeds = embeddingsList.first else {
            throw ZImageTurboLoRATrainerError.datasetEmpty
        }

        // Add batch dimension back: [seqLen, hiddenSize] -> [1, seqLen, hiddenSize]
        return firstEmbeds.expandedDimensions(axis: 0)
    }

    static func encodeTrainingImage(
        _ url: URL,
        vae: AutoencoderKL,
        width: Int,
        height: Int,
        latentHeight: Int,
        latentWidth: Int
    ) throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ZImageTurboLoRATrainerError.imageNotFound(url)
        }

        // Use center crop (ai-toolkit style) to preserve aspect ratio
        let resizedArray = try QwenImageIO.resizedCenterCropPixelArray(
            from: url,
            width: width,
            height: height,
            addBatchDimension: true,
            dtype: .float32
        )

        let normalized = QwenImageIO.normalizeForEncoder(resizedArray)
        let encoded = vae.encode(normalized)

        let latentChannels = vae.configuration.latentChannels
        let mean = encoded[0..., 0..<latentChannels, 0..., 0...]
        let cleanLatent = (mean - MLXArray(vae.configuration.shiftFactor)) * MLXArray(vae.configuration.scalingFactor)

        return cleanLatent.asType(.bfloat16)
    }
}
