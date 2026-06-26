import Foundation
import MLX
import MLXNN
import MLXRandom

extension Flux2KleinLoRATrainer {
    // MARK: - Dataset Encoding

    static func encodePromptEmbeds(
        prompt: String,
        tokenizer: QwenTokenizer,
        textEncoder: QwenTextEncoder,
        maxLength: Int
    ) throws -> MLXArray {
        let promptWithTemplate = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let encoded = tokenizer.encodePlain(prompts: [promptWithTemplate], maxLength: maxLength)

        let result = textEncoder.forwardWithHiddenStates(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask
        )

        guard let hiddenStates = result.hiddenStates, hiddenStates.count >= 28 else {
            throw Flux2Error.insufficientHiddenStates
        }

        let h1 = hiddenStates[9]
        let h2 = hiddenStates[18]
        let h3 = hiddenStates[27]
        return concatenated([h1, h2, h3], axis: -1)
    }

    static func encodeTrainingImage(
        _ url: URL,
        vae: AutoencoderKL,
        width: Int,
        height: Int,
        patchedHeight: Int,
        patchedWidth: Int,
        bnMean: MLXArray,
        bnVar: MLXArray
    ) throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Flux2KleinLoRATrainerError.imageNotFound(url)
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

        let patchified = patchifyLatents(cleanLatent, height: patchedHeight * 2, width: patchedWidth * 2)

        let normalizedPacked = Flux2KleinBatchNorm.normalizePackedLatents(
            patchified,
            mean: bnMean,
            variance: bnVar
        )

        return normalizedPacked
            .transposed(0, 2, 3, 1)
            .reshaped([1, patchedHeight * patchedWidth, 128])
            .asType(.bfloat16)
    }

    private static func patchifyLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        let batch = latents.shape[0]
        let channels = latents.shape[1]

        var x = latents.reshaped([batch, channels, height / 2, 2, width / 2, 2])
        x = x.transposed(0, 1, 3, 5, 2, 4)
        return x.reshaped([batch, channels * 4, height / 2, width / 2])
    }
}
