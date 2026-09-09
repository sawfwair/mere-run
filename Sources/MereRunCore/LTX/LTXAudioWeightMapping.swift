import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func mapAudioVaeDecoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped: String
    if key.hasPrefix("audio_vae.decoder.") {
        mapped = String(key.dropFirst("audio_vae.decoder.".count))
    } else if key == "audio_vae.per_channel_statistics.mean-of-means"
        || key == "audio_vae.per_channel_statistics._mean_of_means" {
        mapped = "per_channel_statistics._mean_of_means"
    } else if key == "audio_vae.per_channel_statistics.std-of-means"
        || key == "audio_vae.per_channel_statistics._std_of_means" {
        mapped = "per_channel_statistics._std_of_means"
    } else {
        return []
    }

    var casted = value
    if sourceLayout == .pytorch,
       mapped.lowercased().contains("conv"),
       mapped.hasSuffix("weight"),
       casted.ndim == 4 {
        casted = casted.transposed(0, 2, 3, 1)
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

func mapAudioVaeEncoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped: String
    if key.hasPrefix("audio_vae.encoder.") {
        mapped = String(key.dropFirst("audio_vae.encoder.".count))
    } else if key == "audio_vae.per_channel_statistics.mean-of-means"
        || key == "audio_vae.per_channel_statistics._mean_of_means" {
        mapped = "per_channel_statistics._mean_of_means"
    } else if key == "audio_vae.per_channel_statistics.std-of-means"
        || key == "audio_vae.per_channel_statistics._std_of_means" {
        mapped = "per_channel_statistics._std_of_means"
    } else {
        return []
    }

    var casted = value
    if sourceLayout == .pytorch,
       mapped.lowercased().contains("conv"),
       mapped.hasSuffix("weight"),
       casted.ndim == 4 {
        casted = casted.transposed(0, 2, 3, 1)
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}
