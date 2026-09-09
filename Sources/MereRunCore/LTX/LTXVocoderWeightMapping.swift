import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func mapVocoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType,
    sourceLayout: LTXVocoderWeightLayout = .pytorch,
    targetFlavor: LTXVocoderFlavor = .legacy
) -> [(String, MLXArray)] {
    guard key.hasPrefix("vocoder.") else { return [] }
    var mapped = String(key.dropFirst("vocoder.".count))
    if targetFlavor == .bandwidthExtension {
        if mapped.hasPrefix("vocoder.") {
            mapped = String(mapped.dropFirst("vocoder.".count))
        }
        if !mapped.hasPrefix("bwe_generator.") && !mapped.hasPrefix("mel_stft.") {
            mapped = "vocoder." + mapped
        }
    }
    mapped = mapped.replacingOccurrences(of: ".downsample.lowpass.filter", with: ".downsample.filter")

    var casted = value
    if sourceLayout == .pytorch {
        if mapped.hasSuffix("weight"), casted.ndim == 3 {
            if mapped.contains("ups.") {
                casted = casted.transposed(1, 2, 0)
            } else {
                casted = casted.transposed(0, 2, 1)
            }
        } else if casted.ndim == 3,
                  mapped.hasSuffix("filter")
                    || mapped.hasSuffix("forward_basis")
                    || mapped.hasSuffix("inverse_basis") {
            casted = casted.transposed(0, 2, 1)
        }
    }
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}
