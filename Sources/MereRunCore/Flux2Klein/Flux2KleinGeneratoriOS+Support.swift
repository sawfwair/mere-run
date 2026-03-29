import Foundation
import MLX
import MLXRandom
import MLXNN
import ImageIO

extension Flux2KleinGeneratoriOS {

    // MARK: - Latent Utilities

    func unpatchifyPackedLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        let batch = latents.shape[0]
        let numChannels = latents.shape[1]

        var x = latents.reshaped([batch, numChannels / 4, 2, 2, height, width])
        x = x.transposed(0, 1, 4, 2, 5, 3)
        x = x.reshaped([batch, numChannels / 4, height * 2, width * 2])

        return x
    }

    func patchifyLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        let batch = latents.shape[0]
        let numChannels = latents.shape[1]

        var x = latents.reshaped([batch, numChannels, height / 2, 2, width / 2, 2])
        x = x.transposed(0, 1, 3, 5, 2, 4)
        x = x.reshaped([batch, numChannels * 4, height / 2, width / 2])

        return x
    }


}
