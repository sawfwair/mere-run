import MLX
import MLXNN

struct MuseGlimmerDFlash2Selection {
    let tokens: MLXArray
    let candidateTokenIds: MLXArray
    let candidateProbabilities: MLXArray?
}

struct MuseGlimmerAssistantDraftOutput {
    let hidden: MLXArray
    let logits: MLXArray
}

enum MuseGlimmerDFlash2WeightKeys {
    static func normalized(_ key: String) -> String {
        switch key {
        case "fc.weight":
            return "encoder.fc.weight"
        case "hidden_norm.weight":
            return "encoder.output_norm_enc.weight"
        case "candidate_selector.predecessor_codebook":
            return "candidate_selector.predecessor_codebook.weight"
        case "candidate_selector.successor_codebook":
            return "candidate_selector.successor_codebook.weight"
        default:
            return key
        }
    }

    static func mapped(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        [(normalized(key), value)]
    }
}

func museGlimmerGroupedDynamicConvolution(
    hidden: MLXArray,
    dynamic: MLXArray,
    base: MLXArray,
    groupSize: Int
) -> MLXArray {
    let batch = hidden.dim(0)
    let length = hidden.dim(1)
    let hiddenSize = hidden.dim(2)
    let groups = hiddenSize / groupSize
    let blocks = hidden.reshaped(batch, length, groups, groupSize)
    let kernels = dynamic.reshaped(batch, length, base.dim(0), groups, 1)
    var output = MLXArray.zeros(like: blocks)

    for offset in 0..<base.dim(0) {
        let values: MLXArray
        if offset == 0 {
            values = blocks
        } else if offset < length {
            values = concatenated(
                [
                    MLXArray.zeros([batch, offset, groups, groupSize], dtype: hidden.dtype),
                    blocks[0..., 0..<(length - offset), 0..., 0...],
                ],
                axis: 1
            )
        } else {
            values = MLXArray.zeros(like: blocks)
        }
        let fixed = base[offset, 0...]
            .reshaped(1, 1, groups, groupSize)
            .asType(hidden.dtype)
        output = output + (fixed + kernels[0..., 0..., offset, 0..., 0...]) * values
    }
    return output.reshaped(hidden.shape)
}

final class MuseGlimmerGroupedDynamicCausalConv: Module {
    @ModuleInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    private let groupSize: Int

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        self._baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        self._kernelProjection.wrappedValue = Linear(
            hiddenSize,
            2 * kernelSize * groups,
            bias: false
        )
        super.init()
    }

    func prepare(_ hidden: MLXArray) -> (hidden: MLXArray, dynamic: MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let projected = kernelProjection(hidden).reshaped(
            hidden.dim(0),
            hidden.dim(1),
            2,
            baseKernel.dim(1),
            groups
        )
        return (
            museGlimmerGroupedDynamicConvolution(
                hidden: hidden,
                dynamic: projected[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0, 0..., 0...],
                groupSize: groupSize
            ),
            projected[0..., 0..., 1, 0..., 0...]
        )
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        museGlimmerGroupedDynamicConvolution(
            hidden: hidden,
            dynamic: dynamic,
            base: baseKernel[1, 0..., 0...],
            groupSize: groupSize
        )
    }
}

final class MuseGlimmerDFlash2CandidateSelector: Module {
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear
    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding

    private let topK: Int

    init(vocabularySize: Int, hiddenSize: Int, rank: Int, topK: Int) {
        self.topK = topK
        self._hiddenProjection.wrappedValue = Linear(hiddenSize, rank, bias: false)
        self._predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: vocabularySize,
            dimensions: rank
        )
        self._successorCodebook.wrappedValue = Embedding(
            embeddingCount: vocabularySize,
            dimensions: rank
        )
        super.init()
    }

    func select(
        hidden: MLXArray,
        logits: MLXArray,
        anchorTokenIds: MLXArray,
        temperature: Float
    ) -> MuseGlimmerDFlash2Selection {
        let candidateTokenIds = argPartition(-logits, kth: topK - 1, axis: -1)[
            0...,
            0...,
            0..<topK
        ]
        let unary = takeAlong(logits, candidateTokenIds, axis: -1)
        let projectedHidden = hiddenProjection(hidden)
        var predecessor = anchorTokenIds.asType(.int32).reshaped(anchorTokenIds.dim(0))
        var path: [MLXArray] = []
        var probabilities: [MLXArray] = []
        path.reserveCapacity(hidden.dim(1))
        probabilities.reserveCapacity(hidden.dim(1))

        for position in 0..<hidden.dim(1) {
            let candidates = candidateTokenIds[0..., position, 0...]
            let edgeScores = (
                predecessorCodebook(predecessor).expandedDimensions(axis: 1)
                    * projectedHidden[0..., position, 0...].expandedDimensions(axis: 1)
                    * successorCodebook(candidates)
            ).sum(axis: -1)
            let scores = unary[0..., position, 0...] + edgeScores
            let selected: MLXArray
            if temperature > 0 {
                let distribution = softmax(
                    scores.asType(.float32) / temperature,
                    axis: -1
                )
                selected = categorical(MLX.log(distribution)).asType(.int32)
                probabilities.append(distribution)
            } else {
                selected = argMax(scores, axis: -1).asType(.int32)
            }
            predecessor = takeAlong(
                candidates,
                selected.expandedDimensions(axis: -1),
                axis: -1
            ).squeezed(axis: -1)
            path.append(predecessor)
        }

        return MuseGlimmerDFlash2Selection(
            tokens: stacked(path, axis: 1),
            candidateTokenIds: candidateTokenIds,
            candidateProbabilities: probabilities.isEmpty ? nil : stacked(probabilities, axis: 1)
        )
    }
}
