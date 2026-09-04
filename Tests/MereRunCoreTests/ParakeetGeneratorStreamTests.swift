import AudioCore
import Foundation
import MLX
import XCTest
@testable import AudioSTT

final class ParakeetGeneratorStreamTests: MereRunCoreTestCase {
    func testVectorizedFrontendPadsAudioShorterThanFFTWindow() throws {
        let config = makeConfig()
        let preprocessor = try ParakeetAudioPreprocessor(config: config.preprocessor)

        let features = preprocessor.logMelSpectrogram(from: [])
        MLX.eval(features)

        XCTAssertEqual(features.shape, [1, 1, 4])
        XCTAssertTrue(features.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testPreparedDecodeInstallsCPUAndGPUStreamsAfterGeneratorActorHop() async throws {
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

    func testPreparedMeasuredDecodeReportsPipelineStages() async throws {
        let config = makeConfig()
        let generator = ParakeetGenerator(
            preparedModel: StreamCheckingParakeetModel(config: config),
            audioPreprocessor: try ParakeetAudioPreprocessor(config: config.preprocessor),
            modelConfig: config
        )

        let measured = try await generator.transcribePreparedMeasured(
            samples: Array(repeating: 0.1, count: 800),
            language: "en"
        )

        XCTAssertEqual(measured.result.text, "stream safe")
        XCTAssertEqual(measured.timings.windowCount, 1)
        XCTAssertGreaterThan(measured.timings.featureExtractionSeconds, 0)
        XCTAssertGreaterThan(measured.timings.decoderSeconds, 0)
        XCTAssertGreaterThan(measured.timings.totalSeconds, 0)
    }

    func testBaseModelUsesConfiguredExternalEncoder() throws {
        let model = ParakeetBaseModel(config: makeConfig())
        let externalEncoder = RecordingParakeetEncoder()
        model.externalEncoder = externalEncoder

        let output = try model.encode(MLX.zeros([1, 2, 4]))
        MLX.eval(output.features)

        XCTAssertEqual(externalEncoder.callCount, 1)
        XCTAssertEqual(output.features.shape, [1, 1, 4])
        XCTAssertEqual(output.features.asArray(Float.self), [7, 7, 7, 7])
        XCTAssertEqual(output.lengths, [1])
    }

    func testCoreMLProviderDecodesOverlappedLongAudioWindows() async throws {
        let config = makeConfig(sampleRate: 100, nFFT: 4)
        let generator = ParakeetGenerator(
            preparedModel: WindowedParakeetModel(config: config),
            audioPreprocessor: try ParakeetAudioPreprocessor(config: config.preprocessor),
            modelConfig: config,
            executionProvider: .coreML(artifactURL: URL(fileURLWithPath: "/fixture"))
        )

        let result = try await generator.transcribePrepared(
            samples: Array(repeating: 0.1, count: 3_200),
            language: "en"
        )

        XCTAssertEqual(result.text, "one two three four")
        XCTAssertEqual(result.duration, 32)
        XCTAssertEqual(result.tokenAlignments?.map(\.id), [1, 2, 3, 4])
        XCTAssertEqual(result.tokenAlignments?.map(\.startSeconds), [13, 14, 27, 31])
    }

    func testCoreMLProviderBatchesLongAudioWindowsAtModelPreferredWidth() async throws {
        let config = makeConfig(sampleRate: 100, nFFT: 4)
        let model = BatchedWindowedParakeetModel(config: config)
        let generator = ParakeetGenerator(
            preparedModel: model,
            audioPreprocessor: try ParakeetAudioPreprocessor(config: config.preprocessor),
            modelConfig: config,
            executionProvider: .coreML(artifactURL: URL(fileURLWithPath: "/fixture"))
        )

        let result = try await generator.transcribePrepared(
            samples: Array(repeating: 0.1, count: 3_200),
            language: "en"
        )

        XCTAssertEqual(result.text, "one two three four")
        XCTAssertEqual(result.tokenAlignments?.map(\.id), [1, 2, 3, 4])
    }

    func testCoreMLWindowRangesUseTwoSecondOverlap() {
        XCTAssertEqual(
            ParakeetCoreMLWindowing.sampleRanges(sampleCount: 3_200, sampleRate: 100),
            [0..<1_500, 1_300..<2_800, 2_600..<3_200]
        )
    }

    func testWindowMergeDeduplicatesZeroDurationBoundaryToken() {
        let first = [
            ParakeetAlignedToken(id: 1, text: " boundary", start: 13, duration: 0),
        ]
        let second = [
            ParakeetAlignedToken(id: 1, text: " boundary", start: 13, duration: 0),
            ParakeetAlignedToken(id: 2, text: " next", start: 14, duration: 0.1),
        ]

        let merged = ParakeetAlignment.mergeLongestCommonSubsequence(
            first,
            second,
            overlapDuration: 2
        )

        XCTAssertEqual(merged.map(\.id), [1, 2])
    }

    private func makeConfig(
        sampleRate: Int = 16_000,
        nFFT: Int = 512
    ) -> ParakeetModelConfig {
        let preprocessor = ParakeetPreprocessorConfig(
            sampleRate: sampleRate,
            normalize: "per_feature",
            windowSize: 0.025,
            windowStride: 0.01,
            window: "hann",
            features: 4,
            nFFT: nFFT,
            dither: 0,
            padTo: 0,
            padValue: 0,
            preemph: 0.97
        )
        return ParakeetModelConfig(
            packaging: .completeMLX,
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

private final class RecordingParakeetEncoder: ParakeetExternalEncoder {
    private(set) var callCount = 0

    func encode(_ mel: MLXArray) throws -> ParakeetEncoderOutput {
        callCount += 1
        return ParakeetEncoderOutput(
            features: MLX.full([1, 1, mel.dim(2)], values: MLXArray(Float(7))),
            lengths: [1]
        )
    }
}

private final class StreamCheckingParakeetModel: ParakeetDecodingModel {
    let config: ParakeetModelConfig

    init(config: ParakeetModelConfig) {
        self.config = config
    }

    func decode(_ mel: MLXArray) throws -> [ParakeetAlignedResult] {
        let output = MLX.multiply(MLX.mean(mel), 2, stream: .gpu)
        MLX.eval(output)
        _ = output.item(Float.self)
        return [ParakeetAlignedResult(text: "stream safe", sentences: [])]
    }
}

private final class WindowedParakeetModel: ParakeetDecodingModel {
    let config: ParakeetModelConfig
    private(set) var callCount = 0

    init(config: ParakeetModelConfig) {
        self.config = config
    }

    func decode(_ mel: MLXArray) throws -> [ParakeetAlignedResult] {
        callCount += 1
        let tokens: [ParakeetAlignedToken]
        switch callCount {
        case 1:
            tokens = [token(1, " one", 13), token(2, " two", 14)]
        case 2:
            tokens = [token(1, " one", 0), token(2, " two", 1), token(3, " three", 14)]
        default:
            tokens = [token(3, " three", 1), token(4, " four", 5)]
        }
        return [ParakeetAlignment.sentencesToResult(
            ParakeetAlignment.tokensToSentences(tokens)
        )]
    }

    private func token(
        _ id: Int,
        _ text: String,
        _ start: TimeInterval
    ) -> ParakeetAlignedToken {
        ParakeetAlignedToken(id: id, text: text, start: start, duration: 0.1)
    }
}

private final class BatchedWindowedParakeetModel: ParakeetDecodingModel {
    let config: ParakeetModelConfig
    var preferredWindowBatchSize: Int { 16 }

    init(config: ParakeetModelConfig) {
        self.config = config
    }

    func decode(_ mel: MLXArray) throws -> [ParakeetAlignedResult] {
        []
    }

    func decodeWindows(
        _ mels: [MLXArray],
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult] {
        guard mels.count == 3 else { return [] }
        return [
            result([token(1, " one", 13), token(2, " two", 14)]),
            result([token(1, " one", 0), token(2, " two", 1), token(3, " three", 14)]),
            result([token(3, " three", 1), token(4, " four", 5)]),
        ]
    }

    private func result(_ tokens: [ParakeetAlignedToken]) -> ParakeetAlignedResult {
        ParakeetAlignment.sentencesToResult(
            ParakeetAlignment.tokensToSentences(tokens)
        )
    }

    private func token(
        _ id: Int,
        _ text: String,
        _ start: TimeInterval
    ) -> ParakeetAlignedToken {
        ParakeetAlignedToken(id: id, text: text, start: start, duration: 0.1)
    }
}
