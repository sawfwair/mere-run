import Foundation
import MereRunTensor
import MLX
import MLXFast
import MLXNN
import MLXRandom

final class Gemma4SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    private let groupSize: Int
    private let bits: Int
    private let mode: QuantizationMode

    init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        groupSize: Int = 16,
        bits: Int = 4,
        mode: QuantizationMode = .nvfp4,
        bias: Bool = false,
        quantized: Bool = true
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
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
        self._biases.wrappedValue = nil
        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(x.ndim - 1)

        if x.ndim == 4 && x.dim(2) == topK {
            let flatX = x.reshaped([batchTokens * topK, 1, inputDim])
            let flatIndices = indices.reshaped([batchTokens * topK])
            let output = applyFlat(flatX, indices: flatIndices, sortedIndices: false)
            return output.reshaped([x.dim(0), x.dim(1), topK, output.dim(2)])
        }

        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        return applyGather(expanded, indices: indices, sortedIndices: false)
            .squeezed(axis: -2)
    }

    func applyFlat(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        applyGather(x, indices: indices, sortedIndices: sortedIndices)
    }

    private func applyGather(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        let output: MLXArray
        if let scales {
            output = portableGatherQuantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: indices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
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
            return output + take(bias, indices, axis: 0).expandedDimensions(axis: -2)
        }
        return output
    }
}

final class Gemma4SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4SwitchLinear

    init(config: Gemma4TextConfig, quantized: Bool = true) {
        self._gateProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: max(1, config.moeIntermediateSize),
            numExperts: max(1, config.numExperts),
            quantized: quantized
        )
        self._upProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: max(1, config.moeIntermediateSize),
            numExperts: max(1, config.numExperts),
            quantized: quantized
        )
        self._downProj.wrappedValue = Gemma4SwitchLinear(
            inputDims: max(1, config.moeIntermediateSize),
            outputDims: config.hiddenSize,
            numExperts: max(1, config.numExperts),
            quantized: quantized
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let routeCount = batchTokens * topK
        guard routeCount >= 64 else {
            let up = upProj(x, indices: indices)
            let gate = gateProj(x, indices: indices)
            return downProj(geluApproximate(gate) * up, indices: indices)
        }

        let inputDim = x.dim(x.ndim - 1)
        let flatIndices = indices.reshaped([routeCount])
        let order = argSort(flatIndices, axis: 0)
        let inverseOrder = argSort(order, axis: 0)
        let sortedIndices = take(flatIndices, order, axis: 0)
        let tokenOrder = order.floorDivide(topK)
        let flatInput = x.reshaped([batchTokens, inputDim])
            .take(tokenOrder, axis: 0)
            .reshaped([routeCount, 1, inputDim])

        let up = upProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: true)
        let gate = gateProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: true)
        let output = downProj.applyFlat(
            geluApproximate(gate) * up,
            indices: sortedIndices,
            sortedIndices: true
        ).take(inverseOrder, axis: 0)
        return output.reshaped([x.dim(0), x.dim(1), topK, output.dim(2)])
    }
}

final class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    private let topK: Int
    private let eps: Float
    private let rootSize: Float

    init(config: Gemma4TextConfig) {
        self.topK = max(1, config.topKExperts)
        self.eps = config.rmsNormEps
        self.rootSize = pow(Float(max(1, config.hiddenSize)), -0.5)
        self._proj.wrappedValue = Linear(config.hiddenSize, max(1, config.numExperts), bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([max(1, config.numExperts)])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let normWeight = scale * MLXArray(rootSize).asType(scale.dtype)
        let routedInput = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
        let expertScores = proj(routedInput)
        let k = min(topK, expertScores.dim(-1))
        let indices = argPartition(-expertScores, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        var weights = takeAlong(expertScores, indices, axis: -1)
        weights = softmax(weights, axis: -1)
        weights = weights * take(perExpertScale, indices, axis: 0)
        return (indices, weights)
    }
}

final class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: Gemma4SwitchGLU

    init(config: Gemma4TextConfig) {
        self._switchGLU.wrappedValue = Gemma4SwitchGLU(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let routed = switchGLU(x, indices: indices)
        return (routed * MLX.expandedDimensions(weights, axis: weights.ndim)).sum(axis: -2)
    }
}
