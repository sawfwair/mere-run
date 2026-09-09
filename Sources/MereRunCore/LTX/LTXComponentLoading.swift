import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func loadLTXVideoDecoder(
    weightsURL: URL,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout
) throws -> LTXVideoDecoder {
    let decoder = LTXVideoDecoder(timestepConditioning: false, architecture: .ltx23Split)
    let stats = try SafetensorsStreamingLoader.loadArrays(
        url: weightsURL,
        where: { key in
            key == "latents_mean"
                || key == "latents_std"
                || key == "vae.per_channel_statistics.mean-of-means"
                || key == "vae.per_channel_statistics.std-of-means"
                || key == "vae_decoder.per_channel_statistics.mean-of-means"
                || key == "vae_decoder.per_channel_statistics.std-of-means"
                || key == "vae_decoder.per_channel_statistics.mean"
                || key == "vae_decoder.per_channel_statistics.std"
        },
        dtype: .float32
    )

    if let mean = stats["latents_mean"]
        ?? stats["vae.per_channel_statistics.mean-of-means"]
        ?? stats["vae_decoder.per_channel_statistics.mean-of-means"]
        ?? stats["vae_decoder.per_channel_statistics.mean"] {
        decoder.latentsMean = mean.asType(.float32)
    }
    if let std = stats["latents_std"]
        ?? stats["vae.per_channel_statistics.std-of-means"]
        ?? stats["vae_decoder.per_channel_statistics.std-of-means"]
        ?? stats["vae_decoder.per_channel_statistics.std"] {
        decoder.latentsStd = std.asType(.float32)
    }

    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: decoder,
        dtype: dtype,
        verify: .none,
        include: { key in
            key.hasPrefix("decoder.")
                || key.hasPrefix("vae.decoder.")
                || key.hasPrefix("vae_decoder.")
        },
        mapper: { key, value in
            mapLTXDecoderWeight(key: key, value: value, dtype: dtype, sourceLayout: sourceLayout)
        },
        batchSize: 24
    )
    return decoder
}

func loadLTXVideoUpsampler(
    weightsURL: URL,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout
) throws -> LTXLatentUpsampler {
    let upsampler = LTXLatentUpsampler(inChannels: 128, midChannels: 1_024, numBlocksPerStage: 4)
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: upsampler,
        dtype: dtype,
        verify: .none,
        include: { _ in true },
        mapper: { key, value in
            mapLTXUpsamplerWeight(key: key, value: value, dtype: dtype, sourceLayout: sourceLayout)
        },
        batchSize: 24
    )
    return upsampler
}

func loadLTXTemporalVideoUpsampler(
    weightsURL: URL,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout
) throws -> LTXTemporalLatentUpsampler {
    let upsampler = LTXTemporalLatentUpsampler(inChannels: 128, midChannels: 512, numBlocksPerStage: 4)
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: upsampler,
        dtype: dtype,
        verify: .none,
        include: { _ in true },
        mapper: { key, value in
            mapLTXUpsamplerWeight(key: key, value: value, dtype: dtype, sourceLayout: sourceLayout)
        },
        batchSize: 24
    )
    return upsampler
}

func loadLTXAudioDecoder(
    weightsURL: URL,
    sourceLayout: LTXTensorWeightLayout
) throws -> LTXAudioDecoder {
    let decoder = LTXAudioDecoder()
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: decoder,
        dtype: .float32,
        verify: .none,
        include: { key in
            key.hasPrefix("audio_vae.decoder.")
                || key.hasPrefix("audio_vae.per_channel_statistics.")
        },
        mapper: { key, value in
            mapAudioVaeDecoderWeight(
                key: key,
                value: value,
                dtype: .float32,
                sourceLayout: sourceLayout
            )
        },
        batchSize: 24
    )
    return decoder
}

func loadLTXVocoder(
    weightsURL: URL,
    sourceLayout: LTXVocoderWeightLayout,
    configurationRoot: URL,
    usesPackedConfiguration: Bool
) throws -> LTXAudioVocoderBase {
    let metadata = try SafetensorsStreamingLoader.metadata(url: weightsURL)
    let flavor = detectLTXVocoderFlavor(keys: metadata.keys)
    let vocoder: LTXAudioVocoderBase
    switch flavor {
    case .legacy:
        vocoder = LTXVocoder()
    case .bandwidthExtension:
        let config = usesPackedConfiguration
            ? try loadLTXPackedBWEVocoderConfig(weightsURL: weightsURL)
            : try loadLTXBWEVocoderConfig(modelRoot: configurationRoot)
        guard let config else {
            throw LTXUnifiedAVGeneratorError.bweVocoderConfigMissing(configurationRoot)
        }
        vocoder = LTXVocoderWithBWE(config: config)
    }
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: weightsURL,
        to: vocoder,
        dtype: .float32,
        verify: .none,
        include: { $0.hasPrefix("vocoder.") },
        mapper: { key, value in
            mapVocoderWeight(
                key: key,
                value: value,
                dtype: .float32,
                sourceLayout: sourceLayout,
                targetFlavor: flavor
            )
        },
        batchSize: 24
    )
    return vocoder
}
