import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package final class LagunaSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") package var gateProj: LagunaSwitchLinear
    @ModuleInfo(key: "up_proj") package var upProj: LagunaSwitchLinear
    @ModuleInfo(key: "down_proj") package var downProj: LagunaSwitchLinear
    let decodeNVFP4RowsPerSIMDGroup: Int
    var prefillPairwiseScaleReuseCertified = false
    var prefillDownPairwiseScaleReuseCertified = false

    package init(config: LagunaConfig) {
        self.decodeNVFP4RowsPerSIMDGroup =
            LagunaMoEAccelerationPolicy.decodeRowsPerSIMDGroup(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize,
                topK: config.numExpertsPerToken,
                xsCandidate:
                    LagunaMoEAccelerationPolicy.decodeNVFP4RowsPerSIMDGroup
            )
        self._gateProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.moeIntermediateSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        self._upProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.moeIntermediateSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        self._downProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.moeIntermediateSize,
            outputDimensions: config.hiddenSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        super.init()
    }

    package func callAsFunction(
        _ x: MLXArray,
        indices: MLXArray,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let routeCount = x.dim(0) * x.dim(1) * indices.dim(2)
        if LagunaMoEAccelerationPolicy.sortedRoutingEnabled, routeCount >= 64 {
            return sorted(
                x,
                indices: indices,
                useCustomKernels: useCustomKernels
            )
        }
        return unsorted(
            x,
            indices: indices,
            useCustomKernels: useCustomKernels
        )
    }

    package func sorted(
        _ x: MLXArray,
        indices: MLXArray,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let tokenCount = batch * sequenceLength
        let topK = indices.dim(2)
        let routeCount = tokenCount * topK
        let inputDimensions = x.dim(-1)
        let flatIndices = indices.reshaped([routeCount])
        let order = argSort(flatIndices, axis: 0)
        let stagedRoute =
            useCustomKernels
                && LagunaMoEAccelerationPolicy.rankedPrefillRouteStagingEnabled
                ? RoutedMoERouting.stageRankedLagunaPrefillRoute(
                    x.reshaped([tokenCount, inputDimensions]),
                    flatIndices: flatIndices,
                    order: order,
                    topK: topK
                )
                : nil
        let sortedIndices: MLXArray
        let flatInput: MLXArray
        let stagedInverseOrder: MLXArray?
        if let stagedRoute {
            sortedIndices = stagedRoute.sortedIndices
            flatInput = stagedRoute.sortedInput
            stagedInverseOrder = stagedRoute.inverseOrder
        } else {
            sortedIndices = flatIndices.take(order, axis: 0)
            let tokenOrder = order.floorDivide(topK)
            flatInput = x.reshaped([tokenCount, inputDimensions])
                .take(tokenOrder, axis: 0)
                .reshaped([routeCount, 1, inputDimensions])
            stagedInverseOrder = nil
        }
        let activated: MLXArray
        if useCustomKernels,
           LagunaMoEAccelerationPolicy.fusedSortedNVFP4MoEEnabled,
           sequenceLength >= LagunaMoEAccelerationPolicy.fusedSortedMinimumSequenceLength,
           gateProj.mode == .nvfp4,
           upProj.mode == .nvfp4,
           gateProj.groupSize == upProj.groupSize,
           gateProj.bits == upProj.bits,
           gateProj.biases == nil,
           upProj.biases == nil,
           let gateScales = gateProj.scales,
           let upScales = upProj.scales,
           let fused = RoutedMoERouting.fusedSortedNVFP4SwiGLU(
               flatInput,
               gateWeight: gateProj.weight,
               gateScales: gateScales,
               upWeight: upProj.weight,
               upScales: upScales,
               sortedExpertIndices: sortedIndices,
               groupSize: gateProj.groupSize,
               bits: gateProj.bits,
               pairwiseScaleReuse: prefillPairwiseScaleReuseCertified
           ) {
            activated = fused
        } else {
            let gate = gateProj.applyFlat(
                flatInput,
                indices: sortedIndices,
                sortedIndices: true
            )
            let up = upProj.applyFlat(
                flatInput,
                indices: sortedIndices,
                sortedIndices: true
            )
            activated = MLXNN.silu(gate) * up
        }
        let sortedOutput: MLXArray
        if useCustomKernels,
           LagunaMoEAccelerationPolicy.fusedSortedNVFP4DownEnabled,
           sequenceLength >= LagunaMoEAccelerationPolicy.fusedSortedMinimumSequenceLength,
           downProj.mode == .nvfp4,
           downProj.biases == nil,
           let downScales = downProj.scales,
           let fusedDown = RoutedMoERouting.sortedNVFP4Projection(
               activated,
               weight: downProj.weight,
               scales: downScales,
               sortedExpertIndices: sortedIndices,
               groupSize: downProj.groupSize,
               bits: downProj.bits,
               pairwiseScaleReuse: prefillDownPairwiseScaleReuseCertified
           ) {
            sortedOutput = fusedDown
        } else {
            sortedOutput = downProj.applyFlat(
                activated,
                indices: sortedIndices,
                sortedIndices: true
            )
        }
        let inverseOrder = stagedInverseOrder
            ?? (
                useCustomKernels
                    && LagunaMoEAccelerationPolicy.fastSortedInverseEnabled
                    ? RoutedMoERouting.invertPermutation(order) ?? argSort(order, axis: 0)
                    : argSort(order, axis: 0)
            )
        return sortedOutput.take(inverseOrder, axis: 0)
            .reshaped([batch, sequenceLength, topK, sortedOutput.dim(-1)])
    }

    package func unsorted(
        _ x: MLXArray,
        indices: MLXArray,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let topK = indices.dim(2)
        if useCustomKernels,
           let fused = lagunaXSDecodeActivation(x, indices: indices) {
            return downProj(
                fused.reshaped([
                    batch,
                    sequenceLength,
                    topK,
                    fused.dim(-1),
                ]),
                indices: indices
            )
        }
        let gate = gateProj(x, indices: indices)
        let up = upProj(x, indices: indices)
        return downProj(MLXNN.silu(gate) * up, indices: indices)
    }

    package func lagunaXSDecodeActivation(
        _ x: MLXArray,
        indices: MLXArray
    ) -> MLXArray? {
        let topK = indices.dim(2)
        guard LagunaMoEAccelerationPolicy.fusedNVFP4MoEEnabled,
           gateProj.mode == .nvfp4,
           upProj.mode == .nvfp4,
           gateProj.groupSize == upProj.groupSize,
           gateProj.bits == upProj.bits,
           gateProj.biases == nil,
           upProj.biases == nil,
           let gateScales = gateProj.scales,
           let upScales = upProj.scales,
           let fused = RoutedMoERouting.fusedGatherNVFP4SwiGLU(
               x,
               gateWeight: gateProj.weight,
               gateScales: gateScales,
               upWeight: upProj.weight,
               upScales: upScales,
               expertIndices: indices,
               topK: topK,
               groupSize: gateProj.groupSize,
               bits: gateProj.bits,
               rowsPerSIMDGroup: decodeNVFP4RowsPerSIMDGroup
           ) else {
            return nil
        }
        return fused
    }

    package func lagunaXSDecodeDownInputs() -> (weight: MLXArray, scales: MLXArray)? {
        guard downProj.mode == .nvfp4,
              downProj.groupSize == 16,
              downProj.bits == 4,
              downProj.biases == nil,
              let scales = downProj.scales,
              downProj.weight.dtype == .uint32,
              downProj.weight.shape == [256, 2_048, 64],
              scales.dtype == .uint8,
              scales.shape == [256, 2_048, 32] else {
            return nil
        }
        return (downProj.weight, scales)
    }

    package func prepareSortedDownWarmUp() -> MLXArray? {
        guard LagunaMoEAccelerationPolicy.fusedSortedNVFP4DownEnabled,
              downProj.mode == .nvfp4,
              downProj.biases == nil,
              let scales = downProj.scales else {
            return nil
        }
        let routeCount = LagunaMoEAccelerationPolicy.fusedSortedMinimumSequenceLength
        let inputDimensions = downProj.weight.dim(2) * 8
        let input = MLXArray.zeros(
            [routeCount, 1, inputDimensions],
            dtype: .bfloat16
        )
        let indices = MLXArray.zeros([routeCount], dtype: .int32)
        return RoutedMoERouting.sortedNVFP4Projection(
            input,
            weight: downProj.weight,
            scales: scales,
            sortedExpertIndices: indices,
            groupSize: downProj.groupSize,
            bits: downProj.bits,
            pairwiseScaleReuse: prefillDownPairwiseScaleReuseCertified
        )
    }

    /// Certifies the loaded routed scale planes once, before warmup. The
    /// results are retained as Booleans only; unlike the challenge runtime's
    /// packed banks, production needs no additional scale storage.
    package func preparePrefillPairwiseScaleReuse() {
        prefillPairwiseScaleReuseCertified = false
        prefillDownPairwiseScaleReuseCertified = false
        guard LagunaMoEAccelerationPolicy.prefillExpertPairwiseScaleReuseEnabled else {
            return
        }
        if gateProj.mode == .nvfp4,
           upProj.mode == .nvfp4,
           gateProj.groupSize == 16,
           upProj.groupSize == 16,
           gateProj.bits == 4,
           upProj.bits == 4,
           gateProj.biases == nil,
           upProj.biases == nil,
           let gateScales = gateProj.scales,
           let upScales = upProj.scales,
           gateScales.dtype == .uint8,
           upScales.dtype == .uint8,
           gateScales.shape == upScales.shape {
            prefillPairwiseScaleReuseCertified =
                lagunaNVFP4AdjacentScalePairsCertified(gateScales)
                && lagunaNVFP4AdjacentScalePairsCertified(upScales)
        }
        if downProj.mode == .nvfp4,
           downProj.groupSize == 16,
           downProj.bits == 4,
           downProj.biases == nil,
           let downScales = downProj.scales,
           downScales.dtype == .uint8 {
            prefillDownPairwiseScaleReuseCertified =
                lagunaNVFP4AdjacentScalePairsCertified(downScales)
        }
    }
}
