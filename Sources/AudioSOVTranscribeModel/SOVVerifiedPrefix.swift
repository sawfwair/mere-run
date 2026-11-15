import MLX
import MereRunGemmaModel

extension SOVTranscribeDecoder {
    /// New KV rows are causal and independent of later rejected proposals.
    /// Append their accepted prefix to the untouched base, preserving history
    /// that a sliding trial may already have evicted from its left edge.
    package static func commitVerifiedPrefix(
        _ cache: [Gemma4AttentionCache], from trial: [Gemma4AttentionCache],
        proposedCount: Int, committedCount: Int
    ) -> Bool {
        precondition(cache.count == trial.count && committedCount > 0 && committedCount <= proposedCount)
        var updates: [(MLXArray, MLXArray)] = []
        for entry in trial {
            guard let state = entry.currentState(), state.0.dim(2) >= proposedCount else { return false }
            let start = state.0.dim(2) - proposedCount
            updates.append((state.0[0..., 0..., start..<(start + committedCount), 0...],
                            state.1[0..., 0..., start..<(start + committedCount), 0...]))
        }
        // Collect every slice before mutating any base entry, so an unsupported
        // short window leaves the entire base available for replay.
        for (entry, state) in zip(cache, updates) { entry.append(keys: state.0, values: state.1) }
        return true
    }
}
