import Foundation
import MLX

package struct DFlash2TreeProposal {
    package let tokens: [Int]
    package let parents: [Int]
    package let logProbabilities: [Double]
    package let copyRows: Set<Int>

    package init(tokens: [Int], parents: [Int], logProbabilities: [Double], copyRows: Set<Int> = []) {
        self.tokens = tokens
        self.parents = parents
        self.logProbabilities = logProbabilities
        self.copyRows = copyRows
    }

    package func prefix(nodes: Int) -> Self {
        precondition((0..<tokens.count).contains(nodes))
        return Self(tokens: Array(tokens.prefix(nodes + 1)), parents: Array(parents.prefix(nodes + 1)),
                    logProbabilities: Array(logProbabilities.prefix(nodes + 1)), copyRows: copyRows.filter { $0 <= nodes })
    }
}

package struct DFlash2TreeCalibration: Sendable {
    package let temperature: Float
    package let edgeWeight: Float
    package static let reference = Self(temperature: 1, edgeWeight: 1)
    package static let candidate = Self(temperature: 1.5, edgeWeight: 0.6)
}

extension DFlash2CandidateSelector {
    /// A parent-closed best-first tree from the unchanged masked draft block.
    /// Each edge uses its actual parent's learned predecessor codebook.
    package func tree(hidden: MLXArray, logits: MLXArray, anchor: Int, nodeBudget: Int,
                      calibration: DFlash2TreeCalibration = .reference,
                      vocabulary: MLXArray? = nil) -> DFlash2TreeProposal {
        precondition(hidden.dim(0) == 1 && (1...15).contains(nodeBudget))
        let count = candidateCount
        let indices = argPartition(-logits, kth: count - 1, axis: -1)[0..., 0..., 0..<count]
        let unary = takeAlong(logits, indices, axis: -1)
        let candidates = vocabulary.map { take($0, indices, axis: 0) } ?? indices
        return tree(hidden: hidden, candidates: candidates, unary: unary, anchor: anchor,
                    nodeBudget: nodeBudget, calibration: calibration)
    }

    package func tree(hidden: MLXArray, candidates: MLXArray, unary: MLXArray, anchor: Int, nodeBudget: Int,
                      calibration: DFlash2TreeCalibration) -> DFlash2TreeProposal {
        let count = candidateCount
        let depth = hidden.dim(1)
        let ids = candidates[0, 0..., 0...].asType(.int32)
        let root = MLXArray(Array(repeating: Int32(anchor), count: count)).reshaped(1, count)
        let parents = depth == 1 ? root : concatenated([root, ids[0..<(depth - 1), 0...]], axis: 0)
        // Score every depth and actual predecessor in one lattice. Separate
        // depth dispatches repeat embedding, product, and reduction overhead.
        let edge = (predecessorCodebook(parents).expandedDimensions(axis: 2)
            * hiddenProjection(hidden).reshaped(depth, 1, 1, -1)
            * successorCodebook(ids).expandedDimensions(axis: 1)).sum(axis: -1)
        let conditional: MLXArray
        if calibration.edgeWeight == 1 && calibration.temperature == 1 {
            conditional = (edge + unary[0, 0..., 0...].expandedDimensions(axis: 1)).asType(.float32)
        } else {
            conditional = ((edge * calibration.edgeWeight
                + unary[0, 0..., 0...].expandedDimensions(axis: 1)) / calibration.temperature).asType(.float32)
        }
        let scores = conditional - logSumExp(conditional, axis: -1, keepDims: true)
        MLX.eval(candidates, scores)
        return Self.buildTree(anchor: anchor, candidates: candidates.asArray(Int32.self).map(Int.init),
            scores: scores.asArray(Float.self), depthCount: hidden.dim(1), candidateCount: count, nodeBudget: nodeBudget)
    }

    private struct FrontierNode {
        let parent: Int
        let depth: Int
        let choice: Int
        let token: Int
        let score: Double
    }

    package static func buildTree(anchor: Int, candidates: [Int], scores: [Float], depthCount: Int,
                                  candidateCount: Int, nodeBudget: Int) -> DFlash2TreeProposal {
        precondition(candidates.count == depthCount * candidateCount)
        precondition(scores.count == depthCount * candidateCount * candidateCount)
        var frontier: [FrontierNode] = []
        func children(parent: Int, depth: Int, previousChoice: Int, score: Double) -> [FrontierNode] {
            guard depth < depthCount else { return [] }
            return (0..<candidateCount).map { choice in
                FrontierNode(parent: parent, depth: depth, choice: choice,
                    token: candidates[depth * candidateCount + choice],
                    score: score + Double(scores[(depth * candidateCount + previousChoice) * candidateCount + choice]))
            }
        }
        frontier = children(parent: 0, depth: 0, previousChoice: 0, score: 0)
        var tokens = [anchor], parents = [-1]
        var logProbabilities = [0.0]
        while tokens.count <= nodeBudget, !frontier.isEmpty {
            let index = frontier.indices.min { lhs, rhs in
                let a = frontier[lhs], b = frontier[rhs]
                if a.score != b.score { return a.score > b.score }
                if a.parent != b.parent { return a.parent < b.parent }
                return a.token < b.token
            }!
            let node = frontier.remove(at: index)
            let parent = tokens.count
            tokens.append(node.token)
            parents.append(node.parent)
            logProbabilities.append(node.score)
            frontier += children(parent: parent, depth: node.depth + 1, previousChoice: node.choice, score: node.score)
        }
        return DFlash2TreeProposal(tokens: tokens, parents: parents, logProbabilities: logProbabilities)
    }
}
