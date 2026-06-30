import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

private let q35SwigluCompiled = compile(shapeless: true) { gate, up in
    MLXNN.silu(gate) * up
}

@inline(__always)
private func q35Swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    q35SwigluCompiled(gate, up)
}

final class Q35SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let groupSize: Int
    let bits: Int

    init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        groupSize: Int,
        bits: Int,
        quantized: Bool,
        bias: Bool
    ) {
        self.groupSize = groupSize
        self.bits = bits

        let scale = sqrt(1.0 / Float(max(1, inputDims)))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )
        let groups = max(1, (inputDims + groupSize - 1) / groupSize)
        self._scales.wrappedValue = quantized
            ? MLXArray.zeros([numExperts, outputDims, groups])
            : nil
        self._biases.wrappedValue = quantized
            ? MLXArray.zeros([numExperts, outputDims, groups])
            : nil
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
        let output = applyFlat(flatX, indices: flatIndices, sortedIndices: false)
        let outDim = output.dim(2)
        var reshaped = output.reshaped([batchTokens, topK, outDim])
        reshaped = reshaped.reshaped([x.dim(0), x.dim(1), topK, outDim])

        return reshaped
    }

    func applyFlat(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        let inputDim = x.dim(x.ndim - 1)
        let output: MLXArray
        if let scales {
            let resolved = resolvedQuantization(inputDim: inputDim, scales: scales)
            output = portableGatherQuantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: indices,
                transpose: true,
                groupSize: resolved.groupSize,
                bits: resolved.bits,
                mode: .affine,
                sortedIndices: sortedIndices
            )
        } else {
            output = gatherMM(
                x,
                weight.swappedAxes(-1, -2),
                rhsIndices: indices,
                sortedIndices: sortedIndices
            )
        }

        if let bias {
            return output + bias.take(indices, axis: 0).expandedDimensions(axis: 1)
        }
        return output
    }

    private func resolvedQuantization(inputDim: Int, scales: MLXArray) -> (groupSize: Int, bits: Int) {
        var resolvedBits = bits
        let packedInputDim = weight.dim(weight.ndim - 1)
        let numerator = packedInputDim * 32
        if inputDim > 0, numerator % inputDim == 0 {
            let inferredBits = numerator / inputDim
            if (2...8).contains(inferredBits) {
                resolvedBits = inferredBits
            }
        }

        var resolvedGroupSize = groupSize
        let scaleGroups = scales.dim(scales.ndim - 1)
        if inputDim > 0, scaleGroups > 0, inputDim % scaleGroups == 0 {
            resolvedGroupSize = inputDim / scaleGroups
        }

        return (resolvedGroupSize, resolvedBits)
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
        let quantized = config.quantization != nil

        self._gateProj.wrappedValue = Q35SwitchLinear(
            inputDims: text.hiddenSize,
            outputDims: text.moeIntermediateSize,
            numExperts: text.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )
        self._upProj.wrappedValue = Q35SwitchLinear(
            inputDims: text.hiddenSize,
            outputDims: text.moeIntermediateSize,
            numExperts: text.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )
        self._downProj.wrappedValue = Q35SwitchLinear(
            inputDims: text.moeIntermediateSize,
            outputDims: text.hiddenSize,
            numExperts: text.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let routeCount = batchTokens * topK
        guard routeCount >= 64 else {
            return unsorted(x, indices: indices)
        }

        let inputDim = x.dim(x.ndim - 1)
        let flatIndices = indices.reshaped([routeCount])
        let order = argSort(flatIndices, axis: 0)
        let inverseOrder = argSort(order, axis: 0)
        let sortedIndices = flatIndices.take(order, axis: 0)
        let tokenOrder = order.floorDivide(topK)
        let flatInput = x.reshaped([batchTokens, inputDim])
            .take(tokenOrder, axis: 0)
            .reshaped([routeCount, 1, inputDim])

        let up = upProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: true)
        let gate = gateProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: true)
        let activated = q35Swiglu(gate, up)
        let output = downProj.applyFlat(activated, indices: sortedIndices, sortedIndices: true)
            .take(inverseOrder, axis: 0)

        return output.reshaped([x.dim(0), x.dim(1), topK, output.dim(2)])
    }

    func unsorted(_ x: MLXArray, indices: MLXArray) -> MLXArray {
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

final class Q35FeedForward: Module {
    @ModuleInfo(key: "gate") var gate: Linear?
    @ModuleInfo(key: "switch_mlp") var switchMLP: Q35SwitchGLU?
    @ModuleInfo(key: "shared_expert") var sharedExpert: Q35MLP?
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear?
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "up_proj") var upProj: Linear?
    @ModuleInfo(key: "down_proj") var downProj: Linear?

    private let topK: Int
    private let normTopKProb: Bool
    private let usesMoE: Bool

    init(config: Q35Config) {
        let text = config.textConfig
        self.usesMoE = text.usesMoE
        self.topK = max(1, text.numExpertsPerTok)
        self.normTopKProb = true

        if text.usesMoE {
            self._gate.wrappedValue = Linear(text.hiddenSize, text.numExperts, bias: false)
            self._switchMLP.wrappedValue = Q35SwitchGLU(config: config)
            self._sharedExpert.wrappedValue = Q35MLP(
                hiddenSize: text.hiddenSize,
                intermediateSize: text.sharedExpertIntermediateSize
            )
            self._sharedExpertGate.wrappedValue = Linear(text.hiddenSize, 1, bias: false)
            self._gateProj.wrappedValue = nil
            self._upProj.wrappedValue = nil
            self._downProj.wrappedValue = nil
        } else {
            self._gate.wrappedValue = nil
            self._switchMLP.wrappedValue = nil
            self._sharedExpert.wrappedValue = nil
            self._sharedExpertGate.wrappedValue = nil
            self._gateProj.wrappedValue = Linear(text.hiddenSize, text.intermediateSize, bias: false)
            self._upProj.wrappedValue = Linear(text.hiddenSize, text.intermediateSize, bias: false)
            self._downProj.wrappedValue = Linear(text.intermediateSize, text.hiddenSize, bias: false)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesMoE,
           let gate,
           let switchMLP,
           let sharedExpert,
           let sharedExpertGate {
            var scores = softmax(gate(x), axis: -1, precise: true)

            let indices = argPartition(scores, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
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

        guard let gateProj, let upProj, let downProj else {
            return x
        }
        return downProj(q35Swiglu(gateProj(x), upProj(x)))
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
        var scores = softmax(gate(x), axis: -1, precise: true)

        let indices = argPartition(scores, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
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
