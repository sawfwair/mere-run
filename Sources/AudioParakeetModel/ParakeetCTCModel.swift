import Foundation
import MLX
import MLXFast
import MLXNN

class ParakeetCTCModel: ParakeetBaseModel {
    @ModuleInfo(key: "decoder") var decoder: ParakeetConvASRDecoder

    override init(config: ParakeetModelConfig, includeMLXEncoder: Bool = true) {
        self._decoder.wrappedValue = ParakeetConvASRDecoder(config: config.ctcDecoder ?? ParakeetCTCDecoderConfig(
            featIn: config.encoder.modelDim,
            numClasses: max(1, config.vocabulary.count),
            vocabulary: config.vocabulary
        ))
        super.init(config: config, includeMLXEncoder: includeMLXEncoder)
    }

    override func decode(_ mel: MLXArray) throws -> [ParakeetAlignedResult] {
        let batch = normalizeBatch(mel)
        let encoderOutput = try encode(batch)
        let encoded = encoderOutput.features
        let lengths = encoderOutput.lengths
        let logits = decoder(encoded)
        MLX.eval(logits)

        var results: [ParakeetAlignedResult] = []
        results.reserveCapacity(batch.dim(0))

        let vocabulary = config.vocabulary
        let blank = vocabulary.count
        let stepSeconds = timePerEncoderStep

        for b in 0..<batch.dim(0) {
            let featureLength = min(lengths[b], logits.dim(1))
            let predictions = logits[b, 0..<featureLength, 0...]
            let bestTokens = MLX.argMax(predictions, axis: 1)

            var hypothesis: [ParakeetAlignedToken] = []
            var tokenBoundaries: [(Int, Int?)] = []
            var previousToken = -1

            for t in 0..<featureLength {
                let token = Int(bestTokens[t].item(Int32.self))
                if token == blank {
                    continue
                }
                if token == previousToken {
                    continue
                }

                if previousToken != -1, let previousStart = tokenBoundaries.last?.0 {
                    let start = TimeInterval(previousStart) * stepSeconds
                    let end = TimeInterval(t) * stepSeconds
                    hypothesis.append(
                        ParakeetAlignedToken(
                            id: previousToken,
                            text: ParakeetTokenizer.decode(tokens: [previousToken], vocabulary: vocabulary),
                            start: start,
                            duration: max(0, end - start)
                        )
                    )
                }

                tokenBoundaries.append((t, nil))
                previousToken = token
            }

            if previousToken != -1, let previousStart = tokenBoundaries.last?.0 {
                var lastNonBlank = max(0, featureLength - 1)
                if featureLength > 1 {
                    for t in stride(from: featureLength - 1, through: previousStart, by: -1) {
                        let token = Int(bestTokens[t].item(Int32.self))
                        if token != blank {
                            lastNonBlank = t
                            break
                        }
                    }
                }

                let start = TimeInterval(previousStart) * stepSeconds
                let end = TimeInterval(lastNonBlank + 1) * stepSeconds
                hypothesis.append(
                    ParakeetAlignedToken(
                        id: previousToken,
                        text: ParakeetTokenizer.decode(tokens: [previousToken], vocabulary: vocabulary),
                        start: start,
                        duration: max(0, end - start)
                    )
                )
            }

            let sentences = ParakeetAlignment.tokensToSentences(hypothesis)
            results.append(ParakeetAlignment.sentencesToResult(sentences))
        }

        return results
    }
}
