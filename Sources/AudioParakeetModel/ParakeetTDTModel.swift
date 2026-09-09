import Foundation
import MLX
import MLXFast
import MLXNN

package class ParakeetTDTModel: ParakeetBaseModel {
    @ModuleInfo(key: "decoder") var decoder: ParakeetPredictNetwork
    @ModuleInfo(key: "joint") var joint: ParakeetJointNetwork
    package var externalDecoder: (any ParakeetExternalTDTDecoder)?

    let durations: [Int]
    let maxSymbols: Int?

    package override var preferredWindowBatchSize: Int {
        externalDecoder?.maximumBatchSize ?? 1
    }

    override init(config: ParakeetModelConfig, includeMLXEncoder: Bool = true) {
        self.durations = config.tdtDurations ?? [0, 1, 2, 3, 4]
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
            numExtraOutputs: max(1, (config.tdtDurations ?? [0, 1]).count)
        ))

        super.init(config: config, includeMLXEncoder: includeMLXEncoder)
    }

    package override func decode(_ mel: MLXArray) throws -> [ParakeetAlignedResult] {
        var timings = ParakeetModelTimings()
        return try decode(mel, timings: &timings)
    }

    package override func decode(
        _ mel: MLXArray,
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult] {
        let batch = normalizeBatch(mel)
        let encoderStarted = ParakeetMonotonicClock.now()
        let encoderOutput = try encode(batch)
        let encoded = encoderOutput.features
        let lengths = encoderOutput.lengths
        MLX.eval(encoded)
        timings.encoderSeconds += ParakeetMonotonicClock.seconds(since: encoderStarted)

        if let externalDecoder {
            let decoderStarted = ParakeetMonotonicClock.now()
            let result = try externalDecoder.decode(encoded: encoded, lengths: lengths)
            timings.decoderSeconds += ParakeetMonotonicClock.seconds(since: decoderStarted)
            return result
        }

        var results: [ParakeetAlignedResult] = []
        results.reserveCapacity(batch.dim(0))

        let vocabularySize = config.vocabulary.count
        let stepSeconds = timePerEncoderStep
        let windowSize = ParakeetDecodingEnvironment.windowedDecodeFrames

        for b in 0..<batch.dim(0) {
            let decoderStarted = ParakeetMonotonicClock.now()
            let features = encoded[b..<(b + 1), 0..., 0...]
            let maxLength = min(lengths[b], features.dim(1))

            var lastToken = vocabularySize
            var tokens: [ParakeetAlignedToken] = []
            var time = 0
            var newSymbols = 0
            var state: (MLXArray, MLXArray)?

            // Greedy TDT decode over windows of frames. The decoder state
            // only changes when a token is emitted, so the joint can be
            // evaluated for many contiguous frames against the same state
            // in one batched call with a single readback; the host then
            // follows the duration jumps until an emission invalidates the
            // window. The legacy loop read two argMax scalars back per
            // frame. Semantics are identical to the per-frame loop.
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

                let classCount = jointOutput.dim(3)
                let predictions = MLX.argMax(
                    jointOutput[0, 0..., 0, 0..<(vocabularySize + 1)],
                    axis: -1
                ).asType(.int32)
                let hasDurations = classCount > vocabularySize + 1
                let readback: [Int32]
                if hasDurations {
                    let decisions = MLX.argMax(
                        jointOutput[0, 0..., 0, (vocabularySize + 1)..<classCount],
                        axis: -1
                    ).asType(.int32)
                    readback = MLX.concatenated([predictions, decisions], axis: 0).asArray(Int32.self)
                } else {
                    readback = predictions.asArray(Int32.self)
                }
                let windowLength = windowEnd - windowStart
                var emitted = false

                while time < windowEnd {
                    let row = time - windowStart
                    let predictedToken = Int(readback[row])
                    let decisionIndex = hasDurations ? Int(readback[windowLength + row]) : 1
                    let clampedDecision = min(max(0, decisionIndex), max(0, durations.count - 1))
                    let durationSteps = max(0, durations[clampedDecision])

                    if predictedToken != vocabularySize {
                        let start = TimeInterval(time) * stepSeconds
                        let duration = TimeInterval(durationSteps) * stepSeconds
                        tokens.append(
                            ParakeetAlignedToken(
                                id: predictedToken,
                                text: tokenText(predictedToken),
                                start: start,
                                duration: duration
                            )
                        )
                        lastToken = predictedToken
                        state = proposedState
                        emitted = true
                    }

                    time += durationSteps
                    newSymbols += 1

                    if durationSteps != 0 {
                        newSymbols = 0
                    } else if let maxSymbols, maxSymbols <= newSymbols {
                        time += 1
                        newSymbols = 0
                    }

                    if emitted {
                        break
                    }
                    if durationSteps == 0, maxSymbols == nil {
                        // Blank with zero duration and no symbol cap would
                        // re-read the same frame forever in the legacy loop
                        // too; preserve its behavior by re-evaluating.
                        break
                    }
                }
            }

            timings.decoderSeconds += ParakeetMonotonicClock.seconds(since: decoderStarted)
            let alignmentStarted = ParakeetMonotonicClock.now()
            let sentences = ParakeetAlignment.tokensToSentences(tokens)
            results.append(ParakeetAlignment.sentencesToResult(sentences))
            timings.alignmentSeconds += ParakeetMonotonicClock.seconds(since: alignmentStarted)
        }

        return results
    }

    package override func decodeWindows(
        _ mels: [MLXArray],
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult] {
        guard let externalDecoder else {
            return try super.decodeWindows(mels, timings: &timings)
        }
        guard !mels.isEmpty else { return [] }
        guard mels.count <= externalDecoder.maximumBatchSize else {
            throw ParakeetError.decoderBatchTooLarge(
                actual: mels.count,
                maximum: externalDecoder.maximumBatchSize
            )
        }

        let encoderStarted = ParakeetMonotonicClock.now()
        var encodedWindows: [MLXArray] = []
        var lengths: [Int] = []
        encodedWindows.reserveCapacity(mels.count)
        lengths.reserveCapacity(mels.count)
        for mel in mels {
            let batch = normalizeBatch(mel)
            guard batch.dim(0) == 1 else {
                throw ParakeetCoreMLError.unsupportedInputShape(batch.shape)
            }
            let encoderOutput = try encode(batch)
            encodedWindows.append(encoderOutput.features)
            lengths.append(encoderOutput.lengths[0])
        }
        let encoded = MLX.concatenated(encodedWindows, axis: 0)
        MLX.eval(encoded)
        timings.encoderSeconds += ParakeetMonotonicClock.seconds(since: encoderStarted)

        let decoderStarted = ParakeetMonotonicClock.now()
        let results = try externalDecoder.decode(encoded: encoded, lengths: lengths)
        timings.decoderSeconds += ParakeetMonotonicClock.seconds(since: decoderStarted)
        return results
    }
}

class ParakeetTDTCTCModel: ParakeetTDTModel {
    @ModuleInfo(key: "ctc_decoder") var ctcDecoder: ParakeetConvASRDecoder

    init(
        config: ParakeetModelConfig,
        ctcConfig: ParakeetCTCDecoderConfig?,
        includeMLXEncoder: Bool = true
    ) {
        self._ctcDecoder.wrappedValue = ParakeetConvASRDecoder(config: ctcConfig ?? ParakeetCTCDecoderConfig(
            featIn: config.encoder.modelDim,
            numClasses: max(1, config.vocabulary.count),
            vocabulary: config.vocabulary
        ))
        super.init(config: config, includeMLXEncoder: includeMLXEncoder)
    }
}
