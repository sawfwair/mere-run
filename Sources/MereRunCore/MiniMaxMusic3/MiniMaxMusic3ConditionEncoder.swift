import MLX
import MLXNN

public final class MiniMaxMusic3ConditionEncoder: Module {
    public let configuration: MiniMaxMusic3ConditionConfiguration

    @ModuleInfo(key: "layer_weight_logits") var layerWeightLogits: MLXArray
    @ModuleInfo(key: "layer_scale") var layerScale: MLXArray
    @ModuleInfo(key: "proj") var projection: Conv1d

    public init(configuration: MiniMaxMusic3ConditionConfiguration) {
        self.configuration = configuration
        self._layerWeightLogits.wrappedValue = MLXArray.zeros([configuration.numConditionLayers])
        self._layerScale.wrappedValue = MLXArray.ones([1])
        self._projection.wrappedValue = Conv1d(
            inputChannels: configuration.conditionHiddenDim,
            outputChannels: configuration.outDim,
            kernelSize: 3,
            padding: 1
        )
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let frames = hiddenStates.dim(1)
        let layers = configuration.numConditionLayers
        var hidden = hiddenStates.reshaped(batch, frames, layers, configuration.conditionHiddenDim)
        let weights = softmax(layerWeightLogits.asType(hidden.dtype), axis: 0).reshaped(1, 1, layers, 1)
        hidden = MLX.sum(hidden * weights, axis: 2) * layerScale.asType(hidden.dtype)
        hidden = projection(hidden)

        let length = MiniMaxMusic3Prompt.latentLength(
            frameCount: frames,
            inputSamplingRate: configuration.inputSamplingRate,
            inputHopLength: configuration.inputHopLength,
            outputSamplingRate: configuration.outputSamplingRate,
            outputHopLength: configuration.outputHopLength
        )
        let indices = MLXArray((0..<length).map { Int32($0 * frames / length) })
        return MLX.take(hidden, indices, axis: 1)
    }

    static func mapWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "proj.weight" {
            return [(key, value.transposed(0, 2, 1))]
        }
        return [(key, value)]
    }
}
