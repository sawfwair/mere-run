import MLX

package enum Flux2LatentPacking {
    // MARK: - Latent Utilities

    /// Unpatchify packed latents from [batch, 128, H, W] to [batch, 32, H*2, W*2]
    /// This matches mflux's _unpatchify_latents
    package static func unpatchifyPackedLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        // Input: [batch, 128, height, width]
        // Output: [batch, 32, height*2, width*2]
        let batch = latents.shape[0]
        let numChannels = latents.shape[1]  // 128

        // Reshape: [batch, 128, H, W] -> [batch, 32, 2, 2, H, W]
        var x = latents.reshaped([batch, numChannels / 4, 2, 2, height, width])

        // Transpose to interleave the 2x2 patches: [batch, 32, H, 2, W, 2]
        x = x.transposed(0, 1, 4, 2, 5, 3)

        // Reshape to final: [batch, 32, H*2, W*2]
        x = x.reshaped([batch, numChannels / 4, height * 2, width * 2])

        return x
    }

    /// Patchify latents from [batch, 32, H, W] to [batch, 128, H/2, W/2]
    /// This is the inverse of unpatchifyPackedLatents
    package static func patchifyLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        // Input: [batch, 32, height, width]
        // Output: [batch, 128, height/2, width/2]
        let batch = latents.shape[0]
        let numChannels = latents.shape[1]  // 32

        // Reshape: [batch, 32, H, W] -> [batch, 32, H/2, 2, W/2, 2]
        var x = latents.reshaped([batch, numChannels, height / 2, 2, width / 2, 2])

        // Transpose to group patches: [batch, 32, 2, 2, H/2, W/2]
        x = x.transposed(0, 1, 3, 5, 2, 4)

        // Reshape to packed: [batch, 128, H/2, W/2]
        x = x.reshaped([batch, numChannels * 4, height / 2, width / 2])

        return x
    }

}
