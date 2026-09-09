import MLX

extension MiniMaxH3AudioVAE {
    static func mapConvertedWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "latents_mean" || key == "latents_std" {
            return [(key, value)]
        }
        let mappedKey = key
        let plainConvolutionPrefixes = ["encoder.", "mean_proj.", "logs_proj.", "dec_in_proj."]
        if plainConvolutionPrefixes.contains(where: mappedKey.hasPrefix) {
            if mappedKey.hasSuffix(".weight"), value.ndim == 3 {
                return [(mappedKey, value.transposed(0, 2, 1))]
            }
            return [(mappedKey, value)]
        }
        if mappedKey.hasPrefix("encoder.") || mappedKey.hasPrefix("pre_block.") {
            return [(mappedKey, value)]
        }
        guard mappedKey.hasPrefix("decoder.") else { return [] }
        let suffix = String(mappedKey.dropFirst("decoder.".count))
        return MMAudioBigVGAN.mapSafetensorsWeights(key: suffix, value: value).map {
            ("decoder.\($0.0)", $0.1)
        }
    }
}
