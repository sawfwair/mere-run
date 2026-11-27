import MLX

extension Inkling2Model {
    /// Keep the packed projection and add the adapter to real vocabulary rows before softcapping.
    package func makeOutputHead() throws -> (MLXArray, ((MLXArray) -> MLXArray)?) -> MLXArray {
        let project: (MLXArray) -> MLXArray
        let key = prefix + "unembed.weight"
        if let q8OutputHead {
            project = { q8OutputHead($0.asType(.bfloat16)) }
        } else if let affine = try weights.referenceProjection(key) {
            project = affine.callAsFunction
        } else {
            let weight = try weights.read(key)
            project = { matmul($0.asType(weight.dtype), weight.T) }
        }
        let config = config.textConfig
        return { hidden, delta in
            var logits = project(hidden).asType(.float32)[.ellipsis, ..<config.unpaddedVocabSize]
            if let delta { logits = logits + delta(hidden) }
            return Inkling2Math.softcap(logits, config.finalLogitSoftcapping)
        }
    }

    /// Each example has its own recurrent cache. Bound prefill workspace as in chat.
    package func frozenHiddenStates(_ inputIDs: MLXArray) throws -> MLXArray {
        let length = inputIDs.dim(1)
        let chunkSize = prefillChunkSize(for: length)
        return try concatenated((0..<inputIDs.dim(0)).map { row in
            let cache = makeCache()
            var chunks: [MLXArray] = []
            for start in stride(from: 0, to: length, by: chunkSize) {
                let end = min(start + chunkSize, length)
                let ids = inputIDs[row..<(row + 1), start..<end]
                let hidden = try hiddenStates(embeddings: embeddings(ids), cache: cache)
                eval([hidden] + cache.layers.flatMap(\.arrays))
                chunks.append(hidden)
            }
            return concatenated(chunks, axis: 1)
        }, axis: 0)
    }
}
