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

enum Q35FusedSwitchGLUPolicy {
    /// Kill switch: MERERUN_Q35_FUSED_SWITCH_GLU=0 disables the stacked
    /// gate+up expert matmul. The fusion halves the gather-matmul launches per
    /// MoE block. The loader materializes this stack one layer at a time and
    /// releases the two source arrays before preparing the next layer.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_Q35_FUSED_SWITCH_GLU"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()
}

class Q35SwitchLinear: Module {
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

        if quantized {
            let packedInputDims = (inputDims * bits + 31) / 32
            self._weight.wrappedValue = MLXArray.zeros(
                [numExperts, outputDims, packedInputDims],
                dtype: .uint32
            )
        } else {
            let scale = sqrt(1.0 / Float(max(1, inputDims)))
            self._weight.wrappedValue = MLXRandom.uniform(
                low: -scale,
                high: scale,
                [numExperts, outputDims, inputDims]
            )
        }
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

    init(
        weight: MLXArray,
        scales: MLXArray?,
        biases: MLXArray?,
        bias: MLXArray?,
        groupSize: Int,
        bits: Int
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self._weight.wrappedValue = weight
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        self._bias.wrappedValue = bias
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
            let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
            return applyGather(expanded, indices: indices, sortedIndices: false).squeezed(axis: -2)
        }

        let flatIndices = indices.reshaped([batchTokens * topK])
        let output = applyFlat(flatX, indices: flatIndices, sortedIndices: false)
        let outDim = output.dim(2)
        var reshaped = output.reshaped([batchTokens, topK, outDim])
        reshaped = reshaped.reshaped([x.dim(0), x.dim(1), topK, outDim])

        return reshaped
    }

    func applyFlat(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        applyGather(x, indices: indices, sortedIndices: sortedIndices)
    }

    private func applyGather(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
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
            return output + bias.take(indices, axis: 0).expandedDimensions(axis: -2)
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

    /// Gate and up expert weights stacked along the output dimension so each
    /// MoE block issues one gather matmul instead of two. Identity-guarded:
    /// weight loading replaces the parameter arrays, which invalidates the
    /// stack and triggers a lazy rebuild on the next forward.
    private struct FusedGateUp {
        let weight: MLXArray
        let scales: MLXArray?
        let biases: MLXArray?
        let intermediate: Int
        let sourceIDs: [ObjectIdentifier]?
    }

    private var fusedGateUp: FusedGateUp?
    private var fusedGateUpAttempted = false
    private var fusedGateUpSourcesReleased = false

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

        let activated: MLXArray
        if let fused = resolvedFusedGateUp() {
            activated = fusedGateUpGLU(flatInput, fused: fused, indices: sortedIndices, sortedIndices: true)
        } else {
            let up = upProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: true)
            let gate = gateProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: true)
            activated = q35Swiglu(gate, up)
        }
        let output = downProj.applyFlat(activated, indices: sortedIndices, sortedIndices: true)
            .take(inverseOrder, axis: 0)

        return output.reshaped([x.dim(0), x.dim(1), topK, output.dim(2)])
    }

    func unsorted(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let activated: MLXArray
        if let fused = resolvedFusedGateUp() {
            let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
            activated = fusedGateUpGLU(expanded, fused: fused, indices: indices, sortedIndices: false)
                .squeezed(axis: -2)
        } else {
            let up = upProj(x, indices: indices)
            let gate = gateProj(x, indices: indices)
            activated = q35Swiglu(gate, up)
        }
        return downProj(activated, indices: indices)
    }

    private func resolvedFusedGateUp() -> FusedGateUp? {
        guard Q35FusedSwitchGLUPolicy.enabled else { return nil }

        if fusedGateUpSourcesReleased {
            return fusedGateUp
        }

        let gateScales = gateProj.scales
        let upScales = upProj.scales
        var sourceIDs = [ObjectIdentifier(gateProj.weight), ObjectIdentifier(upProj.weight)]
        if let gateScales {
            sourceIDs.append(ObjectIdentifier(gateScales))
        }
        if let upScales {
            sourceIDs.append(ObjectIdentifier(upScales))
        }

        if let fused = fusedGateUp {
            if fused.sourceIDs == sourceIDs { return fused }
            fusedGateUp = nil
            fusedGateUpAttempted = false
        }
        if !fusedGateUpAttempted {
            fusedGateUpAttempted = true
            fusedGateUp = buildFusedGateUp(sourceIDs: sourceIDs)
        }
        return fusedGateUp
    }

    @discardableResult
    func prepareFusedGateUpAndReleaseSources() -> Bool {
        guard Q35FusedSwitchGLUPolicy.enabled else { return false }
        if fusedGateUpSourcesReleased {
            return fusedGateUp != nil
        }

        guard let built = buildFusedGateUp(sourceIDs: nil) else {
            fusedGateUpAttempted = true
            return false
        }

        fusedGateUp = built
        fusedGateUpAttempted = true
        Stream.gpu.synchronize()
        let releasedGate = Q35SwitchLinear(
            weight: MLXArray.zeros([0], dtype: gateProj.weight.dtype),
            scales: nil,
            biases: nil,
            bias: nil,
            groupSize: gateProj.groupSize,
            bits: gateProj.bits
        )
        let releasedUp = Q35SwitchLinear(
            weight: MLXArray.zeros([0], dtype: upProj.weight.dtype),
            scales: nil,
            biases: nil,
            bias: nil,
            groupSize: upProj.groupSize,
            bits: upProj.bits
        )
        update(modules: ModuleChildren.unflattened([
            ("gate_proj", releasedGate),
            ("up_proj", releasedUp),
        ]))
        fusedGateUpSourcesReleased = true
        return true
    }

    var hasPreparedFusedGateUp: Bool {
        fusedGateUp != nil && fusedGateUpSourcesReleased
    }

    var sourceGateUpElementCount: Int {
        gateProj.weight.shape.reduce(1, *) + upProj.weight.shape.reduce(1, *)
    }

    private func buildFusedGateUp(sourceIDs: [ObjectIdentifier]?) -> FusedGateUp? {
        guard gateProj.bias == nil, upProj.bias == nil else { return nil }
        guard gateProj.weight.ndim == 3,
              gateProj.weight.shape == upProj.weight.shape,
              gateProj.weight.dtype == upProj.weight.dtype else {
            return nil
        }

        let quantizationMatches = (gateProj.scales == nil) == (upProj.scales == nil)
            && (gateProj.biases == nil) == (upProj.biases == nil)
        guard quantizationMatches else { return nil }
        if let gateScales = gateProj.scales, let upScales = upProj.scales {
            guard gateScales.shape == upScales.shape else { return nil }
        }

        let weight = concatenated([gateProj.weight, upProj.weight], axis: 1)
        let scales = gateProj.scales.flatMap { gateScales in
            upProj.scales.map { concatenated([gateScales, $0], axis: 1) }
        }
        let biases = gateProj.biases.flatMap { gateBiases in
            upProj.biases.map { concatenated([gateBiases, $0], axis: 1) }
        }
        var toEvaluate = [weight]
        if let scales { toEvaluate.append(scales) }
        if let biases { toEvaluate.append(biases) }
        MLX.eval(toEvaluate)

        return FusedGateUp(
            weight: weight,
            scales: scales,
            biases: biases,
            intermediate: gateProj.weight.dim(1),
            sourceIDs: sourceIDs
        )
    }

    /// One gather matmul over the stacked weights, split into gate/up, with the
    /// swiglu applied as raw ops (the split output makes the compiled swiglu
    /// helper's extra call overhead pointless here).
    private func fusedGateUpGLU(
        _ x: MLXArray,
        fused: FusedGateUp,
        indices: MLXArray,
        sortedIndices: Bool
    ) -> MLXArray {
        let inputDim = x.dim(x.ndim - 1)
        let output: MLXArray
        if let scales = fused.scales {
            let resolved = q35ResolvedQuantization(
                inputDim: inputDim,
                packedInputDim: fused.weight.dim(fused.weight.ndim - 1),
                scaleGroups: scales.dim(scales.ndim - 1),
                fallbackGroupSize: gateProj.groupSize,
                fallbackBits: gateProj.bits
            )
            output = portableGatherQuantizedMM(
                x,
                fused.weight,
                scales: scales,
                biases: fused.biases,
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
                fused.weight.swappedAxes(-1, -2),
                rhsIndices: indices,
                sortedIndices: sortedIndices
            )
        }

        let parts = split(output, indices: [fused.intermediate], axis: -1)
        return MLXNN.silu(parts[0]) * parts[1]
    }
}

private func q35ResolvedQuantization(
    inputDim: Int,
    packedInputDim: Int,
    scaleGroups: Int,
    fallbackGroupSize: Int,
    fallbackBits: Int
) -> (groupSize: Int, bits: Int) {
    var resolvedBits = fallbackBits
    let numerator = packedInputDim * 32
    if inputDim > 0, numerator % inputDim == 0 {
        let inferredBits = numerator / inputDim
        if (2...8).contains(inferredBits) {
            resolvedBits = inferredBits
        }
    }

    var resolvedGroupSize = fallbackGroupSize
    if inputDim > 0, scaleGroups > 0, inputDim % scaleGroups == 0 {
        resolvedGroupSize = inputDim / scaleGroups
    }

    return (resolvedGroupSize, resolvedBits)
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
    private let isQwen4Exp: Bool

    init(config: Q35Config) {
        let text = config.textConfig
        self.usesMoE = text.usesMoE
        self.isQwen4Exp = text.isQwen4Exp
        self.topK = max(1, text.numExpertsPerTok)
        self.normTopKProb = text.normTopKProb

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

    @discardableResult
    func prepareFusedSwitchGLU() -> Bool {
        switchMLP?.prepareFusedGateUpAndReleaseSources() ?? false
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesMoE,
           let gate,
           let switchMLP,
           let sharedExpert,
           let sharedExpertGate {
            let routerLogits = isQwen4Exp ? q38SmallBatchProjection(gate, x) : gate(x)
            // Flash-Next selects and normalizes experts in FP32. Rounding the
            // softmax to BF16 first can turn distinct probabilities into ties.
            var scores = softmax(
                isQwen4Exp ? routerLogits.asType(.float32) : routerLogits,
                axis: -1, precise: true
            )

            let indices = argPartition(scores, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
            scores = takeAlong(scores, indices, axis: -1)

            if normTopKProb, topK > 1 || isQwen4Exp {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }
            if isQwen4Exp { scores = scores.asType(routerLogits.dtype) }

            let switched = switchMLP(x, indices: indices)
            var routed = switched * MLX.expandedDimensions(scores, axis: scores.ndim)
            routed = routed.sum(axis: -2)

            let shared = sharedExpert(x)
            let sharedGate = isQwen4Exp ? q38SmallBatchProjection(sharedExpertGate, x) : sharedExpertGate(x)
            let gatedShared = MLX.sigmoid(sharedGate) * shared
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
        self.normTopKProb = text.normTopKProb

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
