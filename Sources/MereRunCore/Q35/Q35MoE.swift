import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

@inline(__always)
private func q35Swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    MLXNN.silu(gate) * up
}

final class Q35SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let groupSize: Int
    let bits: Int

    init(inputDims: Int, outputDims: Int, numExperts: Int, groupSize: Int, bits: Int, bias: Bool) {
        self.groupSize = groupSize
        self.bits = bits

        let scale = sqrt(1.0 / Float(max(1, inputDims)))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )
        let groups = max(1, (inputDims + groupSize - 1) / groupSize)
        self._scales.wrappedValue = MLXArray.zeros([numExperts, outputDims, groups])
        self._biases.wrappedValue = MLXArray.zeros([numExperts, outputDims, groups])
        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(x.ndim - 1)

        let flatX: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatX = x.reshaped([batchTokens * topK, 1, inputDim])
        } else {
            var expanded = x.reshaped([batchTokens, 1, inputDim])
            expanded = MLX.expandedDimensions(expanded, axis: 1)
            expanded = MLX.repeated(expanded, count: topK, axis: 1)
            flatX = expanded.reshaped([batchTokens * topK, 1, inputDim])
        }

        let flatIndices = indices.reshaped([batchTokens * topK])

        let output: MLXArray
        if let scales {
            output = gatherQuantizedMM(
                flatX,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: flatIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                sortedIndices: false
            )
        } else {
            output = gatherMM(
                flatX,
                weight,
                rhsIndices: flatIndices,
                sortedIndices: false
            )
        }

        let outDim = output.dim(2)
        var reshaped = output.reshaped([batchTokens, topK, outDim])
        reshaped = reshaped.reshaped([x.dim(0), x.dim(1), topK, outDim])

        if let bias {
            return reshaped + bias.reshaped([1, 1, 1, outDim])
        }
        return reshaped
    }
}

final class Q35SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Q35SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: Q35SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: Q35SwitchLinear

    init(config: Q35Config) {
        let text = config.textConfig
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4

        self._gateProj.wrappedValue = Q35SwitchLinear(
            inputDims: text.hiddenSize,
            outputDims: text.moeIntermediateSize,
            numExperts: text.numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )
        self._upProj.wrappedValue = Q35SwitchLinear(
            inputDims: text.hiddenSize,
            outputDims: text.moeIntermediateSize,
            numExperts: text.numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )
        self._downProj.wrappedValue = Q35SwitchLinear(
            inputDims: text.moeIntermediateSize,
            outputDims: text.hiddenSize,
            numExperts: text.numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let up = upProj(x, indices: indices)
        let gate = gateProj(x, indices: indices)
        let activated = q35Swiglu(gate, up)
        return downProj(activated, indices: indices)
    }
}

final class Q35MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(q35Swiglu(gateProj(x), upProj(x)))
    }
}

final class Q35MoE: Module {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: Q35SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Q35MLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    private let topK: Int
    private let normTopKProb: Bool

    init(config: Q35Config) {
        let text = config.textConfig
        self.topK = max(1, text.numExpertsPerTok)
        self.normTopKProb = true

        self._gate.wrappedValue = Linear(text.hiddenSize, text.numExperts, bias: false)
        self._switchMLP.wrappedValue = Q35SwitchGLU(config: config)
        self._sharedExpert.wrappedValue = Q35MLP(
            hiddenSize: text.hiddenSize,
            intermediateSize: text.sharedExpertIntermediateSize
        )
        self._sharedExpertGate.wrappedValue = Linear(text.hiddenSize, 1, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var scores = softmax(gate(x), axis: -1)

        let kth = topK - 1
        let indices = argPartition(-scores, kth: kth, axis: -1)[.ellipsis, 0..<topK]
        scores = takeAlong(scores, indices, axis: -1)

        if normTopKProb, topK > 1 {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let switched = switchMLP(x, indices: indices)
        var routed = switched * MLX.expandedDimensions(scores, axis: scores.ndim)
        routed = routed.sum(axis: -2)

        let shared = sharedExpert(x)
        let gatedShared = MLX.sigmoid(sharedExpertGate(x)) * shared
        return routed + gatedShared
    }
}
