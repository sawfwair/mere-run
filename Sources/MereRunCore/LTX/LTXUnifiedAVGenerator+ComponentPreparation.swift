import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXUnifiedAVGenerator {
    func installRuntimeLoRAsIfNeeded(
        user: [LTXLoRAConfiguration],
        detailing: [LTXLoRAConfiguration]
    ) throws {
        guard !user.isEmpty || !detailing.isEmpty else { return }
        guard let transformer else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        if runtimeUserLoRAAdapters.isEmpty {
            runtimeUserLoRAAdapters = try user.map { configuration in
                guard FileManager.default.fileExists(atPath: configuration.url.path) else {
                    throw LTXUnifiedAVGeneratorError.loraMissing(configuration.url)
                }
                return try LTXRuntimeLoRAAdapter.install(
                    url: configuration.url,
                    into: transformer,
                    strength: configuration.strength,
                    expectedPairCount: nil
                )
            }
        }
        if runtimeDetailingLoRAAdapters.isEmpty {
            runtimeDetailingLoRAAdapters = try detailing.map { configuration in
                guard FileManager.default.fileExists(atPath: configuration.url.path) else {
                    throw LTXUnifiedAVGeneratorError.loraMissing(configuration.url)
                }
                return try LTXRuntimeLoRAAdapter.install(
                    url: configuration.url,
                    into: transformer,
                    strength: configuration.strength,
                    expectedPairCount: nil
                )
            }
        }
    }

    func loadEncoderIfNeeded() throws {
        if encoder != nil {
            return
        }
        guard let modelWeightsURL else {
            throw LTXUnifiedAVGeneratorError.encoderNotLoaded
        }
        let encoderURL = videoEncoderWeightsURL ?? modelWeightsURL

        let vaeEncoder = LTXVideoEncoder(architecture: videoVAEArchitecture)
        if let decoder {
            vaeEncoder.latentsMean = decoder.latentsMean.asType(.float32)
            vaeEncoder.latentsStd = decoder.latentsStd.asType(.float32)
        } else {
            let stats = try SafetensorsStreamingLoader.loadArrays(
                url: encoderURL,
                where: { key in
                    key == "latents_mean"
                        || key == "latents_std"
                        || key == "vae.per_channel_statistics.mean-of-means"
                        || key == "vae.per_channel_statistics.std-of-means"
                        || key == "vae_encoder.per_channel_statistics.mean-of-means"
                        || key == "vae_encoder.per_channel_statistics.std-of-means"
                        || key == "vae_encoder.per_channel_statistics._mean_of_means"
                        || key == "vae_encoder.per_channel_statistics._std_of_means"
                },
                dtype: .float32
            )
            if let mean = stats["latents_mean"]
                ?? stats["vae.per_channel_statistics.mean-of-means"]
                ?? stats["vae_encoder.per_channel_statistics.mean-of-means"]
                ?? stats["vae_encoder.per_channel_statistics._mean_of_means"] {
                vaeEncoder.latentsMean = mean.asType(.float32)
            }
            if let std = stats["latents_std"]
                ?? stats["vae.per_channel_statistics.std-of-means"]
                ?? stats["vae_encoder.per_channel_statistics.std-of-means"]
                ?? stats["vae_encoder.per_channel_statistics._std_of_means"] {
                vaeEncoder.latentsStd = std.asType(.float32)
            }
        }

        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: encoderURL,
            to: vaeEncoder,
            dtype: loadedDType,
            verify: .none,
            include: { key in
                key.hasPrefix("encoder.")
                    || key.hasPrefix("vae.encoder.")
                    || key.hasPrefix("vae_encoder.")
            },
            mapper: { key, value in
                mapLTXEncoderWeight(
                    key: key,
                    value: value,
                    dtype: loadedDType,
                    sourceLayout: videoVAEWeightLayout
                )
            },
            batchSize: 24
        )

        encoder = vaeEncoder
    }
}
