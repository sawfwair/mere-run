import MLX

/// Applies repetition to the configured history window, then additive penalties
/// to generated tokens only. The caller's logits remain unchanged.
func applyingSamplingPenalties(
    _ logits: MLXArray,
    config: GenerationConfig,
    history: MLXArray?
) -> MLXArray {
    guard let history, history.size > 0 else { return logits }
    var result = logits
    if let penalty = config.repetitionPenalty, penalty != 1, config.repetitionContextSize > 0 {
        let start = max(0, history.size - config.repetitionContextSize)
        let indices = history[start...]
        let selected = logits[indices]
        let penalized = MLX.where(selected .< 0, selected * penalty, selected / penalty)
        // Index assignment mutates an MLXArray wrapper, so use a separate wrapper.
        result = logits + MLXArray.zeros(like: logits)
        result[indices] = penalized
    }
    if (config.presencePenalty != 0 || config.frequencyPenalty != 0),
       history.size > config.penaltyPromptTokenCount {
        let generated = history[config.penaltyPromptTokenCount...]
        let counts = MLXArray.zeros([logits.dim(-1)], dtype: .float32)
            .at[generated].add(MLXArray.ones([generated.size], dtype: .float32))
        result = result.asType(.float32)
            - (counts .> 0).asType(.float32) * config.presencePenalty
            - counts * config.frequencyPenalty
    }
    return result
}
