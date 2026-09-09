import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func mapDistilledTransformerWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("model.diffusion_model.") else {
        return []
    }

    var mapped = String(key.dropFirst("model.diffusion_model.".count))
    mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    mapped = mapped.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    if mapped.hasPrefix("video_embeddings_connector") || mapped.hasPrefix("audio_embeddings_connector") {
        return []
    }
    if mapped.hasPrefix("text_embedding_projection") {
        return []
    }

    var casted = value
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}
