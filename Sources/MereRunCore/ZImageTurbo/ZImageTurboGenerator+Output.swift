import Foundation
import MLX
import MLXNN

extension ZImageTurboGenerator {
    func decodeAndSave(
        latents: MLXArray,
        vae: AutoencoderKL,
        inferenceConfig: ZImageTurboInferenceConfig,
        outputURL: URL,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        progressHandler?(GenerationProgress(
            stage: .decoding,
            stepIndex: inferenceConfig.numInferenceSteps,
            totalSteps: inferenceConfig.numInferenceSteps
        ))

        let decoded = decodeLatents(
            latents,
            vae: vae,
            height: inferenceConfig.height,
            width: inferenceConfig.width
        )

        progressHandler?(GenerationProgress(
            stage: .saving,
            stepIndex: inferenceConfig.numInferenceSteps,
            totalSteps: inferenceConfig.numInferenceSteps
        ))
        try QwenImageIO.saveImage(array: decoded, to: outputURL)
    }

    func ensureOutputDirectory(_ outputURL: URL) throws {
        let dir = outputURL.deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw GeneratorError.invalidOutputDirectory(outputURL)
        }
    }

    func decodeLatents(
        _ latents: MLXArray,
        vae: AutoencoderKL,
        height: Int,
        width: Int
    ) -> MLXArray {
        let (decoded, _) = vae.decode(latents)
        var image = decoded

        if height != decoded.dim(2) || width != decoded.dim(3) {
            var nhwc = image.transposed(0, 2, 3, 1)
            let hScale = Float(height) / Float(decoded.dim(2))
            let wScale = Float(width) / Float(decoded.dim(3))
            nhwc = MLXNN.Upsample(scaleFactor: .array([hScale, wScale]), mode: .nearest)(nhwc)
            image = nhwc.transposed(0, 3, 1, 2)
        }

        image = QwenImageIO.denormalizeFromDecoder(image)
        return MLX.clip(image, min: 0, max: 1)
    }

}
