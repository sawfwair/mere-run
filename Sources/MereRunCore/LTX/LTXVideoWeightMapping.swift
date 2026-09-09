import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func mapLTXDecoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped: String
    if key.hasPrefix("vae.decoder.") {
        mapped = String(key.dropFirst("vae.decoder.".count))
    } else if key.hasPrefix("vae_decoder.") {
        mapped = String(key.dropFirst("vae_decoder.".count))
    } else if key.hasPrefix("decoder.") {
        mapped = String(key.dropFirst("decoder.".count))
    } else {
        return []
    }

    mapped = mapped.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    mapped = mapped.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    var casted = value
    if sourceLayout == .pytorch, mapped.contains(".conv.weight"), casted.ndim == 5 {
        casted = casted.transposed(0, 2, 3, 4, 1)
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }

    return [(mapped, casted)]
}

func mapLTXEncoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped: String
    if key.hasPrefix("vae.encoder.") {
        mapped = String(key.dropFirst("vae.encoder.".count))
    } else if key.hasPrefix("vae_encoder.") {
        mapped = String(key.dropFirst("vae_encoder.".count))
    } else if key.hasPrefix("encoder.") {
        mapped = String(key.dropFirst("encoder.".count))
    } else {
        return []
    }

    var casted = value
    if sourceLayout == .pytorch, mapped.contains(".conv.weight"), casted.ndim == 5 {
        casted = casted.transposed(0, 2, 3, 4, 1)
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }

    return [(mapped, casted)]
}

func mapLTXUpsamplerWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXTensorWeightLayout = .pytorch
) -> [(String, MLXArray)] {
    var mapped = key
    for prefix in [
        "spatial_upscaler_x2_v1_1.",
        "spatial_upscaler_x1_5_v1_0.",
        "temporal_upscaler_x2_v1_0.",
    ] where mapped.hasPrefix(prefix) {
        mapped = String(mapped.dropFirst(prefix.count))
        break
    }
    mapped = mapped.replacingOccurrences(of: "upsampler.0.", with: "upsampler.conv.")

    var casted = value
    if sourceLayout == .pytorch, mapped.contains("conv"), mapped.contains("weight") {
        if casted.ndim == 5 {
            casted = casted.transposed(0, 2, 3, 4, 1)
        } else if casted.ndim == 4 {
            casted = casted.transposed(0, 2, 3, 1)
        }
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

enum LTXTensorWeightLayout: Equatable {
    case pytorch
    case mlx
}
