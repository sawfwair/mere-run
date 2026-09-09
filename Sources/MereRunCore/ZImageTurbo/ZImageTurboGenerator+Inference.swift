import Foundation
import MediaIO
import MLX
import MLXNN

/// Owns prompt encoding and image-to-image preparation.
/// These helpers stay separate from model loading so the inference data flow is
/// easy to trace on its own.
extension ZImageTurboGenerator {
    func encodePrompt(
        _ prompt: String,
        tokenizer: QwenTokenizer,
        textEncoder: QwenTextEncoder,
        maxLength: Int
    ) throws -> MLXArray {
        let encoded = try tokenizer.encodeChat(prompts: [prompt], maxLength: maxLength)
        let embeddingsList = textEncoder.encodeForZImage(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask
        )

        guard let firstEmbeds = embeddingsList.first else {
            throw NSError(domain: "ZImageTurboGenerator", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Text encoder returned empty embeddings."
            ])
        }

        return firstEmbeds.expandedDimensions(axis: 0)
    }

    func prepareImg2ImgLatents(
        inputImageURL: URL,
        noise: MLXArray,
        sigma: MLXArray,
        vae: AutoencoderKL,
        height: Int,
        width: Int
    ) async throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: inputImageURL.path) else {
            throw GeneratorError.inputImageNotFound(inputImageURL)
        }

        let image: MediaImage
        do {
            image = try MediaImageIO.decode(inputImageURL)
        } catch {
            throw GeneratorError.inputImageDecodeFailed(inputImageURL)
        }

        let resizedArray = try QwenImageIO.resizedPixelArray(
            from: image,
            width: width,
            height: height,
            addBatchDimension: true,
            dtype: .float32
        )

        let normalized = QwenImageIO.normalizeForEncoder(resizedArray)
        let encoded = vae.encode(normalized)
        let latentChannels = vae.configuration.latentChannels
        let mean = encoded[0..., 0..<latentChannels, 0..., 0...]
        let cleanFloat = (mean - MLXArray(vae.configuration.shiftFactor)) * MLXArray(vae.configuration.scalingFactor)
        let clean = cleanFloat.asType(noise.dtype)

        let sigmaCast = sigma.asType(noise.dtype)
        let one = MLXArray(1.0).asType(noise.dtype)
        let blended = (one - sigmaCast) * clean + sigmaCast * noise
        MLX.eval(blended)
        return blended
    }
}
