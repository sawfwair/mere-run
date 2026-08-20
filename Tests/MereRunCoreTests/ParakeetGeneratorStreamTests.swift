import AudioCore
import MLX
import XCTest
@testable import AudioSTT

final class ParakeetGeneratorStreamTests: MereRunCoreTestCase {
    func testPreparedDecodeInstallsMLXStreamAfterGeneratorActorHop() async throws {
        let config = makeConfig()
        let generator = ParakeetGenerator(
            preparedModel: StreamCheckingParakeetModel(config: config),
            audioPreprocessor: try ParakeetAudioPreprocessor(config: config.preprocessor),
            modelConfig: config
        )

        let result = try await Task.detached {
            try await Device.withDefaultDevice(.gpu) {
                await Task.yield()
                return try await generator.transcribePrepared(
                    samples: Array(repeating: 0.1, count: 800),
                    language: "en"
                )
            }
        }.value

        XCTAssertEqual(result.text, "stream safe")
        XCTAssertEqual(result.language, "en")
    }

    private func makeConfig() -> ParakeetModelConfig {
        let preprocessor = ParakeetPreprocessorConfig(
            sampleRate: 16_000,
            normalize: "per_feature",
            windowSize: 0.025,
            windowStride: 0.01,
            window: "hann",
            features: 4,
            nFFT: 512,
            dither: 0,
            padTo: 0,
            padValue: 0,
            preemph: 0.97
        )
        return ParakeetModelConfig(
            variant: .ctc,
            target: "test",
            preprocessor: preprocessor,
            encoder: ParakeetEncoderConfig(
                featIn: 4,
                layers: 1,
                modelDim: 4,
                heads: 1,
                ffExpansionFactor: 1,
                subsamplingFactor: 1,
                selfAttentionModel: "abs_pos",
                subsampling: "dw_striding",
                convKernelSize: 3,
                subsamplingConvChannels: 4,
                posEmbMaxLen: 16,
                causalDownsampling: false,
                useBias: true,
                xScaling: false,
                subsamplingConvChunkingFactor: 1
            ),
            rnntDecoder: nil,
            ctcDecoder: ParakeetCTCDecoderConfig(
                featIn: 4,
                numClasses: 1,
                vocabulary: ["stream"]
            ),
            joint: nil,
            tdtDurations: nil,
            maxSymbols: nil,
            quantizationBits: nil,
            quantizationGroupSize: nil,
            supportedLanguageCodes: ["en"]
        )
    }
}

private final class StreamCheckingParakeetModel: ParakeetDecodingModel {
    let config: ParakeetModelConfig

    init(config: ParakeetModelConfig) {
        self.config = config
    }

    func decode(_ mel: MLXArray) -> [ParakeetAlignedResult] {
        let output = MLX.mean(mel) * 2
        MLX.eval(output)
        _ = output.item(Float.self)
        return [ParakeetAlignedResult(text: "stream safe", sentences: [])]
    }
}
