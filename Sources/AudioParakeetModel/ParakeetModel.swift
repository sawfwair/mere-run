import Foundation
import MLX
import MLXFast
import MLXNN

package class ParakeetBaseModel: Module, ParakeetDecodingModel {
    package let config: ParakeetModelConfig
    @ModuleInfo(key: "encoder") var encoder: ParakeetConformer?
    package var externalEncoder: (any ParakeetExternalEncoder)?

    package var preferredWindowBatchSize: Int { 1 }

    init(config: ParakeetModelConfig, includeMLXEncoder: Bool = true) {
        self.config = config
        self._encoder.wrappedValue = includeMLXEncoder
            ? ParakeetConformer(config: config.encoder)
            : nil
    }

    package func decode(_ mel: MLXArray) throws -> [ParakeetAlignedResult] {
        []
    }

    package func decode(
        _ mel: MLXArray,
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult] {
        let started = ParakeetMonotonicClock.now()
        let result = try decode(mel)
        timings.decoderSeconds += ParakeetMonotonicClock.seconds(since: started)
        return result
    }

    package func decodeWindows(
        _ mels: [MLXArray],
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult] {
        try mels.map { mel in
            try decode(mel, timings: &timings).first
                ?? ParakeetAlignedResult(text: "", sentences: [])
        }
    }

    func encode(_ mel: MLXArray) throws -> ParakeetEncoderOutput {
        if let externalEncoder {
            return try externalEncoder.encode(mel)
        }
        guard let encoder else {
            throw ParakeetError.missingMLXEncoder
        }
        let (features, lengths) = encoder(mel)
        return ParakeetEncoderOutput(features: features, lengths: lengths)
    }

    var timePerEncoderStep: TimeInterval {
        TimeInterval(config.encoder.subsamplingFactor * config.preprocessor.hopLength)
            / TimeInterval(config.preprocessor.sampleRate)
    }

    func normalizeBatch(_ mel: MLXArray) -> MLXArray {
        if mel.ndim == 2 {
            return mel.expandedDimensions(axis: 0)
        }
        return mel
    }

    func tokenText(_ token: Int) -> String {
        ParakeetTokenizer.decode(tokens: [token], vocabulary: config.vocabulary)
    }
}

public enum ParakeetModelFactory {
    public static func build(
        config: ParakeetModelConfig,
        includeMLXEncoder: Bool = true
    ) -> any ParakeetDecodingModel {
        switch config.variant {
        case .tdt:
            return ParakeetTDTModel(config: config, includeMLXEncoder: includeMLXEncoder)
        case .tdtCTC:
            return ParakeetTDTCTCModel(
                config: config,
                ctcConfig: config.ctcDecoder,
                includeMLXEncoder: includeMLXEncoder
            )
        case .rnnt:
            return ParakeetRNNTModel(config: config, includeMLXEncoder: includeMLXEncoder)
        case .ctc:
            return ParakeetCTCModel(config: config, includeMLXEncoder: includeMLXEncoder)
        }
    }
}
