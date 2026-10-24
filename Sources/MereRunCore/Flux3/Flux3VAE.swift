import Foundation
import MLX
import MLXNN

final class Flux3VAE {
    static var configuration: VAEConfig {
        VAEConfig(latentChannels: 32, scalingFactor: 1, shiftFactor: 0,
            blockOutChannels: [128, 256, 512, 512], useQuantConv: true, usePostQuantConv: true, attentionNormEps: 1e-6)
    }
    let model: AutoencoderKL
    let mean: MLXArray
    let variance: MLXArray

    convenience init(resources: Flux3Resources) throws {
        try self.init(arrays: MLX.loadArrays(url: resources.rootURL.appending(path: "ae.safetensors")))
    }

    init(configuration: VAEConfig = Flux3VAE.configuration, arrays: [String: MLXArray]) throws {
        model = AutoencoderKL(configuration: configuration)
        guard let mean = arrays["bn.running_mean"], let variance = arrays["bn.running_var"],
              mean.shape == [128], variance.shape == [128] else {
            throw Flux3Error.invalidWeights("Missing 128-channel autoencoder batch-normalization statistics.")
        }
        self.mean = mean.asType(.float32)
        self.variance = variance.asType(.float32)
        let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        var mapped: [String: MLXArray] = [:]
        for (key, value) in arrays {
            if ["bn.running_mean", "bn.running_var", "bn.num_batches_tracked"].contains(key) { continue }
            // Both publishers use the original LDM encoder/decoder namespaces.
            let name = Ideogram4VAEWeights.mapKey(key)
            guard let expected = shapes[name], mapped[name] == nil else {
                throw Flux3Error.invalidWeights("Unexpected autoencoder tensor: \(key).")
            }
            let converted: MLXArray
            if value.ndim == 4 && expected.count == 2 && value.dim(2) == 1 && value.dim(3) == 1 {
                converted = value.reshaped(value.dim(0), value.dim(1))
            } else if value.ndim == 4 {
                converted = value.transposed(0, 2, 3, 1)
            } else { converted = value }
            guard converted.shape == expected else { throw Flux3Error.invalidWeights("Invalid autoencoder shape: \(key).") }
            mapped[name] = converted.asType(.float32)
        }
        guard Set(mapped.keys) == Set(shapes.keys) else { throw Flux3Error.invalidWeights("Incomplete autoencoder weights.") }
        try model.update(parameters: ModuleParameters.unflattened(mapped), verify: [.all])
        eval(model, self.mean, self.variance)
    }

    func encode(_ image: MLXArray) -> MLXArray {
        let encoded = model.encode(image)[0..., 0..<32, 0..., 0...]
        return Flux3Latents.pack(encoded, runningMean: mean, runningVariance: variance).asType(.bfloat16)
    }

    func decode(_ tokens: MLXArray, height: Int, width: Int) -> MLXArray {
        let latent = Flux3Latents.unpack(tokens, height: height, width: width, runningMean: mean, runningVariance: variance)
        let pixels = model.decode(latent).0
        return clip((pixels + 1) / 2, min: 0, max: 1)
    }
}
