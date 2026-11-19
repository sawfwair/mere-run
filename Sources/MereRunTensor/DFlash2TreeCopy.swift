import Foundation

extension DFlash2TreeProposal {
    /// Merge a confirmed-context continuation into the proposal frontier. Copy
    /// probabilities are utility priors; only the target authorizes any output.
    package func mergingCopy(_ continuation: [Int], maximumNodes: Int, probability: Double = 0.75) -> Self {
        precondition((1...15).contains(maximumNodes) && probability > 0 && probability < 1)
        var ids = tokens, links = parents
        var own = logProbabilities.indices.map { $0 == 0 ? 0 : logProbabilities[$0] - logProbabilities[parents[$0]] }
        var copied = copyRows
        var parent = 0
        for token in continuation.prefix(maximumNodes) {
            if let row = links.indices.dropFirst().first(where: { links[$0] == parent && ids[$0] == token }) {
                if log(probability) > own[row] { own[row] = log(probability); copied.insert(row) }
                parent = row
            } else {
                ids.append(token); links.append(parent); own.append(log(probability))
                parent = ids.count - 1; copied.insert(parent)
            }
        }
        var scores = [0.0]
        for row in ids.indices.dropFirst() { scores.append(scores[links[row]] + own[row]) }
        var frontier = links.indices.dropFirst().filter { links[$0] == 0 }
        var map = [0: 0], out = [ids[0]], outParents = [-1], outScores = [0.0], outCopies: Set<Int> = []
        while out.count <= maximumNodes && !frontier.isEmpty {
            let index = frontier.indices.min {
                let a = frontier[$0], b = frontier[$1]
                if scores[a] != scores[b] { return scores[a] > scores[b] }
                if map[links[a]] != map[links[b]] { return map[links[a]]! < map[links[b]]! }
                return ids[a] < ids[b]
            }!
            let row = frontier.remove(at: index), next = out.count
            map[row] = next; out.append(ids[row]); outParents.append(map[links[row]]!); outScores.append(scores[row])
            if copied.contains(row) { outCopies.insert(next) }
            frontier += links.indices.dropFirst().filter { links[$0] == row }
        }
        return Self(tokens: out, parents: outParents, logProbabilities: outScores, copyRows: outCopies)
    }
}
