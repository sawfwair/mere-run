import MLX

public enum ZImageTurboLatentCreator {
    /// Creates Z-Image Turbo noise latents in the expected layout: `(C, F, H, W)` where `C=16` and `F=1`.
    public static func createNoise(
        seed: UInt64,
        height: Int,
        width: Int,
        dtype: DType = .bfloat16
    ) -> MLXArray {
        let key = MLX.key(seed)
        let shape = [16, 1, max(height, 8) / 8, max(width, 8) / 8]
        return MLX.normal(shape, dtype: .float32, key: key).asType(dtype)
    }

    /// Converts VAE-encoded latents into the layout expected by the transformer: `(C, F, H, W)`.
    public static func pack(latents: MLXArray) -> MLXArray {
        var x = latents
        if x.ndim == 5 {
            x = x[0..., 0..., 0, 0..., 0...]
        }
        x = x.expandedDimensions(axis: 2)
        x = x.squeezed(axis: 0)
        return x
    }

    /// Converts transformer latents `(C, F, H, W)` back into a VAE-friendly layout `(B, C, H, W)`.
    public static func unpack(latents: MLXArray) -> MLXArray {
        var x = latents
        x = x.expandedDimensions(axis: 0)
        x = x.squeezed(axis: 2)
        return x
    }
}

