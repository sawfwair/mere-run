import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package final class LagunaRouter: Module {
    @ParameterInfo(key: "weight") package var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") package var correctionBias: MLXArray

    let topK: Int
    let expertCount: Int
    let normalize: Bool
    let softcap: Float

    package init(config: LagunaConfig) {
        self.topK = config.numExpertsPerToken
        self.expertCount = config.numExperts
        self.normalize = config.normTopKProbability
        self.softcap = config.moeRouterLogitSoftcapping
        self._weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize])
        self._correctionBias.wrappedValue = MLXArray.zeros([config.numExperts])
        super.init()
    }

    package func callAsFunction(
        _ x: MLXArray,
        useCustomKernels: Bool = true
    ) -> (indices: MLXArray, weights: MLXArray) {
        var logits = x.matmul(weight.T).asType(.float32)
        if softcap > 0 {
            logits = tanh(logits / softcap) * softcap
        }
        if let routed = LagunaActive64Router.routeIfEnabled(
            logits: logits,
            correctionBias: correctionBias,
            expertCount: expertCount,
            topK: topK,
            normalizing: normalize,
            useCustomKernels: useCustomKernels
        ) {
            return (routed.indices, routed.weights.asType(x.dtype))
        }
        let scores = sigmoid(logits)
        let selectionScores = scores + correctionBias.asType(scores.dtype)
        let count = min(topK, selectionScores.dim(-1))
        // Hard expert selection is discrete. Keep gradients through the
        // selected score values, but do not ask gather kernels for an
        // undefined VJP with respect to their integer indices.
        let indices = stopGradient(
            argPartition(-selectionScores, kth: count - 1, axis: -1)[
                .ellipsis,
                ..<count
            ]
        )
        var weights = takeAlong(scores, indices, axis: -1)
        if normalize {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (indices, weights.asType(x.dtype))
    }
}

package final class LagunaSparseMoE: LagunaFeedForward {
    @ModuleInfo(key: "gate") package var gate: LagunaRouter
    @ModuleInfo(key: "switch_mlp") package var switchMLP: LagunaSwitchGLU
    @ModuleInfo(key: "shared_expert") package var sharedExpert: LagunaDenseMLP

    let scalingFactor: Float

    package init(config: LagunaConfig) {
        self._gate.wrappedValue = LagunaRouter(config: config)
        self._switchMLP.wrappedValue = LagunaSwitchGLU(config: config)
        self._sharedExpert.wrappedValue = LagunaDenseMLP(
            inputDimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
        self.scalingFactor = config.moeRoutedScalingFactor
        super.init()
    }

    package override func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, residual: nil)
    }

    package func callAsFunction(
        _ x: MLXArray,
        residual: MLXArray?,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let routed = gate(x, useCustomKernels: useCustomKernels)
        if useCustomKernels,
           LagunaMoEAccelerationPolicy.fusedRoutedSharedDownResidualEnabled,
           let residual,
           scalingFactor == 2.5,
           let routedActivated = switchMLP.lagunaXSDecodeActivation(
               x,
               indices: routed.indices
           ),
           let routedDown = switchMLP.lagunaXSDecodeDownInputs(),
           let sharedDown = sharedExpert.lagunaXSDecodeDownInputs(x),
           let fused = RoutedMoERouting.fusedLagunaXSRoutedSharedDownResidual(
               routedActivated: routedActivated,
               routedDownWeight: routedDown.weight,
               routedDownScales: routedDown.scales,
               indices: routed.indices,
               routerWeights: routed.weights,
               sharedActivated: sharedDown.activated,
               sharedDownWeight: sharedDown.weight,
               sharedDownScales: sharedDown.scales,
               residual: residual
           ) {
            return fused
        }
        var expertOutput = switchMLP(
            x,
            indices: routed.indices,
            useCustomKernels: useCustomKernels
        )
        expertOutput = (
            expertOutput * MLX.expandedDimensions(routed.weights, axis: routed.weights.ndim)
        ).sum(axis: -2)
        if scalingFactor != 1 {
            expertOutput = expertOutput * scalingFactor
        }
        let branch = expertOutput + sharedExpert(x)
        return residual.map { $0 + branch } ?? branch
    }

    package func preparePrefillAcceleration() -> MLXArray? {
        switchMLP.prepareSortedDownWarmUp()
    }

    package func preparePrefillPairwiseScaleReuse() {
        switchMLP.preparePrefillPairwiseScaleReuse()
    }
}
