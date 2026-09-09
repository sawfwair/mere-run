import Foundation
import MLX
import MLXFast
import MLXNN

final class ParakeetLSTMStack: Module {
    @ModuleInfo(key: "lstm") var lstm: [LSTM]

    let hiddenSize: Int

    init(inputSize: Int, hiddenSize: Int, numLayers: Int, bias: Bool) {
        self.hiddenSize = hiddenSize
        self._lstm.wrappedValue = (0..<numLayers).map { index in
            LSTM(
                inputSize: index == 0 ? inputSize : hiddenSize,
                hiddenSize: hiddenSize,
                bias: bias
            )
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        state: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        var output = x
        var nextH: [MLXArray] = []
        var nextC: [MLXArray] = []
        nextH.reserveCapacity(lstm.count)
        nextC.reserveCapacity(lstm.count)

        for index in 0..<lstm.count {
            let hidden = state?.0[index, 0..., 0...]
            let cell = state?.1[index, 0..., 0...]

            let (allH, allC) = lstm[index](output, hidden: hidden, cell: cell)
            output = allH

            let lastStep = max(0, allH.dim(1) - 1)
            nextH.append(allH[0..., lastStep, 0...])
            nextC.append(allC[0..., lastStep, 0...])
        }

        return (
            output,
            (
                MLX.stacked(nextH, axis: 0),
                MLX.stacked(nextC, axis: 0)
            )
        )
    }
}

final class ParakeetPredictNetwork: Module {
    final class Prediction: Module {
        @ModuleInfo(key: "embed") var embed: Embedding
        @ModuleInfo(key: "dec_rnn") var decRNN: ParakeetLSTMStack

        init(config: ParakeetRNNTDecoderConfig) {
            let embeddingSize = config.blankAsPad ? config.vocabSize + 1 : config.vocabSize
            self._embed.wrappedValue = Embedding(
                embeddingCount: embeddingSize,
                dimensions: config.prednet.predHidden
            )

            self._decRNN.wrappedValue = ParakeetLSTMStack(
                inputSize: config.prednet.predHidden,
                hiddenSize: config.prednet.rnnHiddenSize ?? config.prednet.predHidden,
                numLayers: config.prednet.predRnnLayers,
                bias: true
            )
        }
    }

    @ModuleInfo(key: "prediction") var prediction: Prediction

    private let predHidden: Int

    init(config: ParakeetRNNTDecoderConfig) {
        self.predHidden = config.prednet.predHidden
        self._prediction.wrappedValue = Prediction(config: config)
    }

    func callAsFunction(
        _ y: MLXArray?,
        state: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let embedded: MLXArray
        if let y {
            embedded = prediction.embed(y)
        } else {
            let batch = state?.0.dim(1) ?? 1
            embedded = MLX.zeros([batch, 1, predHidden], dtype: prediction.embed.weight.dtype)
        }

        return prediction.decRNN(embedded, state: state)
    }
}

final class ParakeetJointNetwork: Module {
    final class Activation: Module {
        let kind: String

        init(kind: String) {
            self.kind = kind.lowercased()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            switch kind {
            case "relu":
                relu(x)
            case "sigmoid":
                sigmoid(x)
            case "tanh":
                tanh(x)
            default:
                relu(x)
            }
        }
    }

    @ModuleInfo(key: "pred") var pred: Linear
    @ModuleInfo(key: "enc") var enc: Linear
    let joint_net: (Activation, Identity, Linear)

    init(config: ParakeetJointConfig) {
        self._pred.wrappedValue = Linear(
            config.jointnet.predHidden,
            config.jointnet.jointHidden,
            bias: true
        )
        self._enc.wrappedValue = Linear(
            config.jointnet.encoderHidden,
            config.jointnet.jointHidden,
            bias: true
        )
        self.joint_net = (
            Activation(kind: config.jointnet.activation),
            Identity(),
            Linear(
                config.jointnet.jointHidden,
                config.numClasses + 1 + config.numExtraOutputs,
                bias: true
            )
        )
    }

    func callAsFunction(_ encInput: MLXArray, _ predInput: MLXArray) -> MLXArray {
        let encProj = enc(encInput)
        let predProj = pred(predInput)
        let hidden = encProj.expandedDimensions(axis: 2) + predProj.expandedDimensions(axis: 1)
        return joint_net.2(joint_net.1(joint_net.0(hidden)))
    }
}

final class ParakeetConvASRDecoder: Module {
    @ModuleInfo(key: "decoder_layers.0") var projection: Conv1d

    let temperature: Float = 1.0

    init(config: ParakeetCTCDecoderConfig) {
        let classCount = (config.numClasses > 0 ? config.numClasses : config.vocabulary.count) + 1
        self._projection.wrappedValue = Conv1d(
            inputChannels: config.featIn,
            outputChannels: classCount,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        logSoftmax(projection(x) / temperature, axis: -1)
    }
}
