import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public func validatedLTXSigmaSchedule(_ values: [Float]) throws -> [Float] {
    guard values.count >= 2,
          values.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
          zip(values, values.dropFirst()).allSatisfy({ $0 >= $1 }),
          values.first! > 0,
          values.last == 0 else {
        throw LTXUnifiedAVGeneratorError.invalidSigmaSchedule(values)
    }
    return values
}

func isLTXAudioOnlyTransformerWeight(_ key: String) -> Bool {
    guard key.hasPrefix("model.diffusion_model.") else { return false }
    let mapped = String(key.dropFirst("model.diffusion_model.".count))
    if mapped.hasPrefix("audio_patchify_proj.")
        || mapped.hasPrefix("audio_adaln_single.")
        || mapped.hasPrefix("audio_prompt_adaln_single.")
        || mapped.hasPrefix("audio_scale_shift_table")
        || mapped.hasPrefix("audio_norm_out.")
        || mapped.hasPrefix("audio_proj_out.") {
        return true
    }
    guard mapped.hasPrefix("transformer_blocks.") else { return false }
    return mapped.contains(".audio_attn1.")
        || mapped.contains(".audio_attn2.")
        || mapped.contains(".audio_ff.")
        || mapped.hasSuffix(".audio_scale_shift_table")
        || mapped.hasSuffix(".audio_prompt_scale_shift_table")
}
