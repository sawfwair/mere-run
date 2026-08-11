import MLX
import MLXNN

final class NemotronHExperts: Module {
    @ModuleInfo(key: "up_proj") var upProjection: NemotronHNVFP4SwitchLinear
    @ModuleInfo(key: "down_proj") var downProjection: NemotronHNVFP4SwitchLinear

    init(config: NemotronHConfig) {
        self._upProjection.wrappedValue = NemotronHNVFP4SwitchLinear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.moeIntermediateSize,
            expertCount: config.nRoutedExperts
        )
        self._downProjection.wrappedValue = NemotronHNVFP4SwitchLinear(
            inputDimensions: config.moeIntermediateSize,
            outputDimensions: config.hiddenSize,
            expertCount: config.nRoutedExperts
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let projected = upProjection(x, indices: indices)
        let activated = MLX.square(MLX.maximum(projected, MLXArray(0).asType(projected.dtype)))
        return downProjection(activated, indices: indices)
    }
}

final class NemotronHSharedExpert: Module {
    @ModuleInfo(key: "up_proj") var upProjection: NemotronHNVFP4Linear
    @ModuleInfo(key: "down_proj") var downProjection: NemotronHNVFP4Linear

    init(config: NemotronHConfig) {
        self._upProjection.wrappedValue = NemotronHNVFP4Linear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.sharedExpertIntermediateSize
        )
        self._downProjection.wrappedValue = NemotronHNVFP4Linear(
            inputDimensions: config.sharedExpertIntermediateSize,
            outputDimensions: config.hiddenSize
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = upProjection(x)
        let activated = MLX.square(MLX.maximum(projected, MLXArray(0).asType(projected.dtype)))
        return downProjection(activated)
    }
}

final class NemotronHRouter: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    private let topK: Int
    private let scalingFactor: Float
    private let normalize: Bool

    init(config: NemotronHConfig) {
        self._weight.wrappedValue = MLXArray.zeros(
            [config.nRoutedExperts, config.hiddenSize]
        )
        self._correctionBias.wrappedValue = MLXArray.zeros([config.nRoutedExperts])
        self.topK = config.numExpertsPerToken
        self.scalingFactor = config.routedScalingFactor
        self.normalize = config.normTopKProbability
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let scores = MLX.sigmoid(x.asType(.float32).matmul(weight.asType(.float32).T))
        let selection = scores + correctionBias.asType(.float32)
        let indices = stopGradient(
            argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        )
        var selected = takeAlong(scores, indices, axis: -1)
        if normalize {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, (selected * scalingFactor).asType(x.dtype))
    }
}

final class NemotronHMoE: NemotronHMixer {
    @ModuleInfo(key: "gate") var gate: NemotronHRouter
    @ModuleInfo(key: "experts") var experts: NemotronHExperts
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHSharedExpert

    init(config: NemotronHConfig) {
        self._gate.wrappedValue = NemotronHRouter(config: config)
        self._experts.wrappedValue = NemotronHExperts(config: config)
        self._sharedExperts.wrappedValue = NemotronHSharedExpert(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let route = gate(x)
        let routed = (
            experts(x, indices: route.indices)
                * route.weights.expandedDimensions(axis: -1)
        ).sum(axis: -2)
        return routed + sharedExperts(x)
    }

    override func callAsFunction(_ x: MLXArray, cache: NemotronHLayerCache?) -> MLXArray {
        callAsFunction(x)
    }
}
