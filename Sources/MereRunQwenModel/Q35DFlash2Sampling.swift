import MereRunTensor
import MLX
import MLXRandom

/// Core supplies the request's target sampling policy without coupling model
/// computation to request parsing. Nil history avoids penalty work entirely.
package struct Q35DFlash2Sampling {
    package let temperature: Float
    package let history: [Int]?
    package let distribution: (MLXArray, MLXArray?) -> MLXArray

    package init(temperature: Float, history: [Int]?, distribution: @escaping (MLXArray, MLXArray?) -> MLXArray) {
        precondition(temperature > 0)
        self.temperature = temperature
        self.history = history
        self.distribution = distribution
    }

    package static func sample(_ probabilities: MLXArray) -> MLXArray {
        categorical(MLX.log(probabilities)).asType(.int32)
    }

    package static func acceptanceProbability(target: MLXArray, draft: MLXArray) -> MLXArray {
        MLX.minimum(target / MLX.maximum(draft, MLXArray(Float.leastNonzeroMagnitude)), MLXArray(Float(1)))
    }

    /// Sparse q has zero mass outside the selector candidates. Subtracting it
    /// from full-vocabulary p retains replacement tokens outside that shortlist.
    package static func correction(
        target: MLXArray, candidateIDs: MLXArray, candidateProbabilities: MLXArray
    ) -> MLXArray {
        let draft = MLXArray.zeros(like: target)
        draft[candidateIDs] = candidateProbabilities
        let residual = MLX.maximum(target - draft, MLXArray(Float(0)))
        let mass = residual.sum()
        // When p == q rejection has probability zero. Still supply a valid
        // categorical input for the unused correction in the batched graph.
        return MLX.where(mass .> 0, residual / MLX.maximum(mass, MLXArray(Float.leastNonzeroMagnitude)), target)
    }

    /// Reserve the original interleaved random keys, but only draw the replacement
    /// at the first rejection (or the bonus after a fully accepted proposal).
    package func verify(
        logits: MLXArray, selection: DFlash2Selection, history: MLXArray?
    ) -> (accepted: MLXArray, nextToken: MLXArray) {
        let probabilities = selection.candidateProbabilities!
        var prefix = history
        var accepted: [MLXArray] = []
        var targets: [MLXArray] = []
        var keys: [MLXArray] = []
        let count = selection.tokens.dim(1)
        for position in 0..<count {
            let target = distribution(logits[0, position, 0...], prefix)
            let token = selection.tokens[0, position]
            let candidates = selection.candidateTokenIds[0, position, 0...]
            let draft = probabilities[0, position, 0...]
            let draftMass = MLX.where(candidates .== token, draft, MLXArray(Float(0))).sum()
            let probability = Self.acceptanceProbability(target: target[token], draft: draftMass)
            accepted.append((MLXRandom.uniform() .< probability).asType(.int32))
            targets.append(target)
            keys.append(resolve())
            if let previous = prefix { prefix = concatenated([previous, token.reshaped(1)]) }
        }
        targets.append(distribution(logits[0, -1, 0...], prefix))
        keys.append(resolve())
        let flags = stacked(accepted)
        let row = argMin(concatenated([flags, MLXArray([Int32(0)])])).asType(.int32)
        let target = stacked(targets)[row]
        let proposalRow = MLX.minimum(row, MLXArray(Int32(count - 1)))
        let residual = Self.correction(
            target: target, candidateIDs: selection.candidateTokenIds[0, proposalRow, 0...],
            candidateProbabilities: probabilities[0, proposalRow, 0...]
        )
        let distribution = MLX.where(row .== count, target, residual)
        let nextToken = categorical(MLX.log(distribution), key: stacked(keys)[row]).asType(.int32)
        return (flags, nextToken)
    }
}
