import MLX

extension MiniMaxH3VideoVAE {
    static func mapCheckpointWeight(key rawKey: String, value: MLXArray) -> [(String, MLXArray)] {
        if rawKey == "decoder.mask_token" { return [] }
        var key = rawKey
        key = key.replacingOccurrences(of: "decoder.x_embedder.", with: "decoder.proj_in.")
        key = key.replacingOccurrences(of: ".ff.w1.", with: ".ff.linear_in.")
        key = key.replacingOccurrences(of: ".ff.w2.", with: ".ff.linear_out.")
        key = key.replacingOccurrences(of: "encoder.down.", with: "encoder.down_blocks.")
        key = key.replacingOccurrences(of: ".block.", with: ".resnets.")
        key = key.replacingOccurrences(of: ".nin_shortcut.", with: ".conv_shortcut.")
        if key.contains(".downsample.conv.") {
            key = key.replacingOccurrences(of: ".downsample.conv.", with: ".downsamplers.0.conv.")
        }

        if key.contains(".attn.to_qkv.") {
            // The released VAE projects to [head, qkv, headDimension], not
            // [qkv, head, headDimension]. Deinterleave the per-head groups
            // before loading the fused global-QKV MLX Linear module.
            let headCount = 32
            let headDimension = 64
            precondition(value.dim(0) == headCount * 3 * headDimension)
            let trailingShape = Array(value.shape.dropFirst())
            let grouped = value.reshaped([headCount, 3, headDimension] + trailingShape)
            let pieces = MLX.split(grouped, parts: 3, axis: 1).map {
                $0.squeezed(axis: 1).reshaped([headCount * headDimension] + trailingShape)
            }
            return [(key, MLX.concatenated(pieces, axis: 0))]
        }
        if value.ndim == 5 {
            return [(key, value.transposed(0, 2, 3, 4, 1))]
        }
        return [(key, value)]
    }
}
