import Foundation
import MLX
import MLXNN
import MereRunAudioModels

extension MMAudioBigVGAN {
    public static func load(resources: MMAudioModelResources) throws -> MMAudioBigVGAN {
        let model = MMAudioBigVGAN()
        let weightsURL = resources.bigVGANWeightsURL()
        if weightsURL.pathExtension == "safetensors" {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: weightsURL,
                to: model,
                dtype: .float16,
                verify: .none,
                mapper: mapSafetensorsWeights
            )
        } else {
            try MMAudioBigVGANCheckpointLoader.load(url: weightsURL, into: model)
        }
        return model
    }

    static func mapCheckpointWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key.hasSuffix(".filter") {
            return []
        }
        let mappedKey = remapUpsampleKey(key)
        if mappedKey.hasSuffix(".weight_g") {
            let base = String(mappedKey.dropLast(".weight_g".count))
            return [("\(base).parametrizations.weight.original0", value)]
        }
        if mappedKey.hasSuffix(".weight_v"), value.ndim == 3 {
            let base = String(mappedKey.dropLast(".weight_v".count))
            let isTranspose = mappedKey.hasPrefix("ups.")
            let transposed = isTranspose ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            return [(
                "\(base).parametrizations.weight.original1",
                transposed.reshaped(-1).reshaped(transposed.shape)
            )]
        }
        return [(mappedKey, value)]
    }

    private static func remapUpsampleKey(_ key: String) -> String {
        var components = key.split(separator: ".").map(String.init)
        if components.count >= 4, components[0] == "ups", components[2] == "0" {
            components[2] = "convolution"
        }
        return components.joined(separator: ".")
    }

    static func mapSafetensorsWeights(key: String, value: MLXArray) -> [(String, MLXArray)] {
        let mappedKey = remapUpsampleKey(key)
        if mappedKey.hasSuffix(".weight"), value.ndim == 3 {
            let base = String(mappedKey.dropLast(".weight".count))
            let isTranspose = mappedKey.hasPrefix("ups.")
            let transposed = isTranspose ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            let magnitudeAxes = isTranspose ? [0, 1] : [1, 2]
            let magnitude = MLX.sqrt(MLX.sum(transposed * transposed, axes: magnitudeAxes, keepDims: true))
            let reshapedMagnitude = isTranspose
                ? magnitude.reshaped(-1, 1, 1)
                : magnitude
            return [
                ("\(base).parametrizations.weight.original0", reshapedMagnitude),
                ("\(base).parametrizations.weight.original1", transposed),
            ]
        }
        return mapCheckpointWeight(key: mappedKey, value: value)
    }
}
