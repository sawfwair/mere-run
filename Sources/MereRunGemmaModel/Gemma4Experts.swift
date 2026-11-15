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
    var prefillWeight: MLXArray? { prefillCache?.value }
    private var prefillCache: Gemma4ExpandedExpertWeights.Cache?

    private let groupSize: Int
    private let bits: Int
    private let mode: QuantizationMode
    var metalVerification: Gemma4MetalExperts?

    /// Shape-only affine slots for streamed conversion or a native packed checkpoint.
    init(packedInputDims: Int, outputDims: Int, numExperts: Int, groupSize: Int, bits: Int,
         prefillExpander: Gemma4ExpandedExpertWeights? = nil) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = .affine
        let shape = [numExperts, outputDims, packedInputDims * bits / 32]
        let packedWeight = MLXArray.zeros(shape, dtype: .uint32)
        self._weight.wrappedValue = packedWeight
        self._scales.wrappedValue = MLXArray.zeros([numExperts, outputDims, packedInputDims / groupSize], dtype: .bfloat16)
        self._biases.wrappedValue = MLXArray.zeros([numExperts, outputDims, packedInputDims / groupSize], dtype: .bfloat16)
        self._bias.wrappedValue = nil
        self.prefillCache = prefillExpander.map {
            Gemma4ExpandedExpertWeights.Cache(packedWeight, inputWidth: packedInputDims, expander: $0)
        }
        super.init()
    }

    /// Packs a loaded floating-point projection once, outside the inference graph.
    init(packing weight: MLXArray, groupSize: Int, bits: Int = 8, prefillExpander: Gemma4ExpandedExpertWeights? = nil) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = .affine
        let packed = MLX.quantized(weight, groupSize: groupSize, bits: bits)
        self._weight.wrappedValue = packed.wq
        self._scales.wrappedValue = packed.scales
        self._biases.wrappedValue = packed.biases
        self._bias.wrappedValue = nil
        self.prefillCache = prefillExpander.map {
            Gemma4ExpandedExpertWeights.Cache(packed.wq, inputWidth: weight.dim(-1), expander: $0)
        }
        super.init()
        eval(parameters())
        if let prefillWeight { eval(prefillWeight) }
    }

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

    @discardableResult
    override func update(parameters: ModuleParameters, verify: Module.VerifyUpdate, path: [String] = [],
                         modulePath: [String] = []) throws -> Self {
        try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        // Module updates mutate an MLXArray's backing context without replacing
        // its Swift identity. Derived encodings must be invalidated explicitly.
        prefillCache?.invalidate()
        return self
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

    func fusedGateUpActivation(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray? {
        guard let metalVerification, metalVerification.fusedActivation, bias == nil,
              let scales, let biases, supportsMetal(x, sortedIndices: sortedIndices),
              weight.dim(1).isMultiple(of: 16) else { return nil }
        return metalVerification.activation(x, weight: weight, scales: scales, biases: biases, indices: indices)
    }

    private func supportsMetal(_ x: MLXArray, sortedIndices: Bool) -> Bool {
        sortedIndices && groupSize == 64 && bits == 8 && mode == .affine
            && x.ndim == 3 && x.dim(1) == 1 && x.dim(0) <= 128 && x.dim(0) / weight.dim(0) < 4
            && x.dtype == .bfloat16 && scales?.dtype == .bfloat16 && biases?.dtype == .bfloat16
            && x.size <= Int(UInt32.max) && weight.size <= Int(UInt32.max)
            && x.dim(2).isMultiple(of: 64) && weight.dim(1).isMultiple(of: 8)
    }

    private func applyGather(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        let output: MLXArray
        if let scales {
            if let prefillCache, sortedIndices, x.ndim == 3, x.dim(0) / weight.dim(0) >= 4 {
                let expanded = prefillCache.resolve(weight, inputWidth: x.dim(-1))
                output = portableGatherQuantizedMM(x, expanded, scales: scales, biases: biases, rhsIndices: indices,
                    transpose: true, groupSize: groupSize, bits: 8, mode: mode, sortedIndices: true)
            } else if let metalVerification, let biases, supportsMetal(x, sortedIndices: sortedIndices) {
                output = metalVerification(x, weight: weight, scales: scales, biases: biases, indices: indices)
            } else {
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
            }
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
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4SwitchLinear?
    @ModuleInfo(key: "up_proj") var upProj: Gemma4SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Gemma4SwitchLinear?
    @ModuleInfo(key: "down_proj") var downProj: Gemma4SwitchLinear

    var useGEMVForVerification = false

    init(packedConfig config: Gemma4TextConfig, groupSize: Int, bits: Int,
         prefillExpander: Gemma4ExpandedExpertWeights? = nil) {
        self._gateProj.wrappedValue = nil
        self._upProj.wrappedValue = nil
        self._gateUpProj.wrappedValue = Gemma4SwitchLinear(packedInputDims: config.hiddenSize,
            outputDims: 2 * config.moeIntermediateSize, numExperts: config.numExperts,
            groupSize: groupSize, bits: bits, prefillExpander: prefillExpander)
        self._downProj.wrappedValue = Gemma4SwitchLinear(packedInputDims: config.moeIntermediateSize,
            outputDims: config.hiddenSize, numExperts: config.numExperts,
            groupSize: groupSize, bits: bits, prefillExpander: prefillExpander)
        super.init()
    }

    /// Replaces the source modules: no captured BF16 weights survive requantization.
    init(packing source: Gemma4SwitchGLU, groupSize: Int, bits: Int = 8, prefillExpander: Gemma4ExpandedExpertWeights? = nil) {
        self._gateProj.wrappedValue = nil
        self._upProj.wrappedValue = nil
        self._gateUpProj.wrappedValue = Gemma4SwitchLinear(
            packing: concatenated([source.gateProj!.weight, source.upProj!.weight], axis: 1), groupSize: groupSize, bits: bits, prefillExpander: prefillExpander)
        self._downProj.wrappedValue = Gemma4SwitchLinear(packing: source.downProj.weight, groupSize: groupSize, bits: bits, prefillExpander: prefillExpander)
        super.init()
    }

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
        self._gateUpProj.wrappedValue = nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let routeCount = batchTokens * topK
        guard routeCount >= 64 else {
            let gate: MLXArray, up: MLXArray
            if let gateUpProj {
                let pair = gateUpProj(x, indices: indices)
                let half = pair.dim(-1) / 2
                gate = pair[.ellipsis, ..<half]; up = pair[.ellipsis, half...]
            } else {
                gate = gateProj!(x, indices: indices)
                up = upProj!(x, indices: indices)
            }
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

        // Keep the grouped matrix kernel for prefill. GEMV avoids mostly empty
        // 16-row expert tiles in a small verification block, but scales poorly.
        let sortedKernel = !(useGEMVForVerification && batchTokens <= 16)
        let activation: MLXArray
        if let fused = gateUpProj?.fusedGateUpActivation(flatInput, indices: sortedIndices, sortedIndices: sortedKernel) {
            activation = fused
        } else {
            let gate: MLXArray, up: MLXArray
            if let gateUpProj {
                let pair = gateUpProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: sortedKernel)
                let half = pair.dim(-1) / 2
                gate = pair[.ellipsis, ..<half]; up = pair[.ellipsis, half...]
            } else {
                up = upProj!.applyFlat(flatInput, indices: sortedIndices, sortedIndices: sortedKernel)
                gate = gateProj!.applyFlat(flatInput, indices: sortedIndices, sortedIndices: sortedKernel)
            }
            activation = geluApproximate(gate) * up
        }
        let output = downProj.applyFlat(
            activation,
            indices: sortedIndices,
            sortedIndices: sortedKernel
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
        let routedInput = MLXFast.rmsNorm(x.asType(scale.dtype), weight: normWeight, eps: eps)
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
        return (routed * MLX.expandedDimensions(weights, axis: weights.ndim)).asType(x.dtype).sum(axis: -2)
    }
}
