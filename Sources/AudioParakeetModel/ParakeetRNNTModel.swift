import Foundation
import MLX
import MLXFast
import MLXNN

class ParakeetRNNTModel: ParakeetBaseModel {
    @ModuleInfo(key: "decoder") var decoder: ParakeetPredictNetwork
    @ModuleInfo(key: "joint") var joint: ParakeetJointNetwork

    let maxSymbols: Int?

    override init(config: ParakeetModelConfig, includeMLXEncoder: Bool = true) {
        self.maxSymbols = config.maxSymbols

        self._decoder.wrappedValue = ParakeetPredictNetwork(config: config.rnntDecoder ?? ParakeetRNNTDecoderConfig(
            blankAsPad: true,
            vocabSize: max(1, config.vocabulary.count),
            prednet: ParakeetPredictNetConfig(predHidden: 640, predRnnLayers: 2, rnnHiddenSize: nil)
        ))
        self._joint.wrappedValue = ParakeetJointNetwork(config: config.joint ?? ParakeetJointConfig(
            numClasses: max(1, config.vocabulary.count),
            vocabulary: config.vocabulary,
            jointnet: ParakeetJointNetConfig(jointHidden: 640, activation: "relu", encoderHidden: config.encoder.modelDim, predHidden: config.rnntDecoder?.prednet.predHidden ?? 640),
            numExtraOutputs: 0
        ))

        super.init(config: config, includeMLXEncoder: includeMLXEncoder)
    }

    override func decode(_ mel: MLXArray) throws -> [ParakeetAlignedResult] {
        let batch = normalizeBatch(mel)
        let encoderOutput = try encode(batch)
        let encoded = encoderOutput.features
        let lengths = encoderOutput.lengths
        MLX.eval(encoded)

        var results: [ParakeetAlignedResult] = []
        results.reserveCapacity(batch.dim(0))

        let vocabularySize = config.vocabulary.count
        let stepSeconds = timePerEncoderStep

        for b in 0..<batch.dim(0) {
            let features = encoded[b..<(b + 1), 0..., 0...]
            let maxLength = min(lengths[b], features.dim(1))

            var lastToken = vocabularySize
            var tokens: [ParakeetAlignedToken] = []
            var time = 0
            var newSymbols = 0
            var state: (MLXArray, MLXArray)?

            // Greedy RNN-T decode over windows of frames: the joint runs for
            // many contiguous frames against the fixed decoder state in one
            // batched call with a single readback, and the host scans blanks
            // forward until an emission changes the state. The legacy loop
            // read one argMax scalar back per frame. Semantics identical.
            let windowSize = ParakeetDecodingEnvironment.windowedDecodeFrames
            while time < maxLength {
                let tokenInput: MLXArray?
                if lastToken == vocabularySize {
                    tokenInput = nil
                } else {
                    tokenInput = MLXArray([Int32(lastToken)]).reshaped(1, 1)
                }

                let (decoderOutput, proposedState) = decoder(tokenInput, state: state)
                let windowStart = time
                let windowEnd = min(windowStart + windowSize, maxLength)
                let windowFeatures = features[0..., windowStart..<windowEnd, 0...]
                let jointOutput = joint(windowFeatures, decoderOutput.asType(windowFeatures.dtype))
                let predictions = MLX.argMax(jointOutput[0, 0..., 0, 0...], axis: -1)
                    .asType(.int32).asArray(Int32.self)

                while time < windowEnd {
                    let predictedToken = Int(predictions[time - windowStart])

                    if predictedToken != vocabularySize {
                        let start = TimeInterval(time) * stepSeconds
                        tokens.append(
                            ParakeetAlignedToken(
                                id: predictedToken,
                                text: tokenText(predictedToken),
                                start: start,
                                duration: stepSeconds
                            )
                        )
                        lastToken = predictedToken
                        state = proposedState

                        newSymbols += 1
                        if let maxSymbols, maxSymbols <= newSymbols {
                            time += 1
                            newSymbols = 0
                        }
                        break
                    } else {
                        time += 1
                        newSymbols = 0
                    }
                }
            }

            let sentences = ParakeetAlignment.tokensToSentences(tokens)
            results.append(ParakeetAlignment.sentencesToResult(sentences))
        }

        return results
    }
}
