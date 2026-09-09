import MereRunMLXTestSupport
import MLX
import MLXNN
import XCTest
@testable import AudioParakeetModel

final class ParakeetModelTests: MLXTestCase {
    func testRecurrentStateMatchesWholeSequenceAcrossTokenSteps() throws {
        let network = ParakeetLSTMStack(inputSize: 4, hiddenSize: 4, numLayers: 2, bias: true)
        let input = MLXArray((0..<32).map { Float($0) / 32 }, [2, 4, 4])
        let (full, fullState) = network(input)
        var state: (MLXArray, MLXArray)?
        var steps: [MLXArray] = []
        for index in 0..<4 {
            let (output, nextState) = network(input[0..., index..<(index + 1), 0...], state: state)
            steps.append(output)
            state = nextState
        }
        let incremental = MLX.concatenated(steps, axis: 1)
        let finalState = try XCTUnwrap(state)
        eval(full, incremental, fullState.0, fullState.1, finalState.0, finalState.1)
        assertClose(full, incremental)
        assertClose(fullState.0, finalState.0)
        assertClose(fullState.1, finalState.1)
    }

    func testJointFrameBatchMatchesIndividualFrameCalls() {
        let network = ParakeetJointNetwork(config: makeJointConfig())
        let encoder = MLXArray((0..<24).map { Float($0) / 24 }, [2, 3, 4])
        let predictor = MLXArray((0..<16).map { -Float($0) / 16 }, [2, 2, 4])
        let batched = network(encoder, predictor)
        let separate = MLX.concatenated((0..<3).map { index in
            network(encoder[0..., index..<(index + 1), 0...], predictor)
        }, axis: 1)
        eval(batched, separate)
        XCTAssertEqual(batched.shape, [2, 3, 2, 6])
        assertClose(batched, separate)
    }

    func testCTCRespectsEncoderLengthAndPreservesTokenTiming() throws {
        let model = ParakeetCTCModel(config: makeConfig(variant: .ctc), includeMLXEncoder: false)
        try model.decoder.projection.update(parameters: ModuleParameters.unflattened([
            ("weight", MLX.eye(4).reshaped(4, 1, 4)),
            ("bias", MLX.zeros([4]))
        ]), verify: .all)
        let decisions = [3, 0, 0, 1, 1, 3, 2, 2]
        let features = MLX.eye(4)[MLXArray(decisions.map(Int32.init))].reshaped(1, 8, 4)
        model.externalEncoder = FixedParakeetEncoder(features: features, lengths: [6])

        let results = try model.decode(MLX.zeros([1, 8, 4]))
        XCTAssertEqual(results.map(\.text), ["ab"])
        let tokens = try XCTUnwrap(results.first).sentences.flatMap(\.tokens)
        XCTAssertEqual(tokens.map(\.id), [0, 1])
        XCTAssertEqual(tokens[0].start, 0.01, accuracy: 1e-9)
        XCTAssertEqual(tokens[0].end, 0.03, accuracy: 1e-9)
        XCTAssertEqual(tokens[1].start, 0.03, accuracy: 1e-9)
        XCTAssertEqual(tokens[1].end, 0.05, accuracy: 1e-9)
    }

    func testExternalTDTDecoderReceivesWindowsInOrderAndRejectsOversizeBatch() throws {
        let model = ParakeetTDTModel(config: makeConfig(variant: .tdt), includeMLXEncoder: false)
        model.externalEncoder = PassthroughParakeetEncoder()
        let decoder = RecordingParakeetDecoder()
        model.externalDecoder = decoder
        let first = MLX.ones([2, 4])
        let second = MLX.ones([2, 4]) * 2
        var timings = ParakeetModelTimings()
        let results = try model.decodeWindows([first, second], timings: &timings)

        XCTAssertEqual(model.preferredWindowBatchSize, 2)
        XCTAssertEqual(results.map(\.text), ["1", "2"])
        XCTAssertEqual(decoder.lengths, [2, 2])
        XCTAssertEqual(decoder.callCount, 1)
        XCTAssertGreaterThanOrEqual(timings.encoderSeconds, 0)
        XCTAssertGreaterThanOrEqual(timings.decoderSeconds, 0)
        XCTAssertThrowsError(try model.decodeWindows([first, second, first], timings: &timings)) { error in
            guard case ParakeetError.decoderBatchTooLarge(actual: 3, maximum: 2) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
        XCTAssertEqual(decoder.callCount, 1)
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray) {
        XCTAssertLessThan(MLX.max(MLX.abs(actual - expected)).item(Float.self), 1e-6)
    }

    private func makeJointConfig() -> ParakeetJointConfig {
        ParakeetJointConfig(
            numClasses: 3,
            vocabulary: ["▁a", "b", "."],
            jointnet: ParakeetJointNetConfig(jointHidden: 4, activation: "relu", encoderHidden: 4, predHidden: 4),
            numExtraOutputs: 2
        )
    }

    private func makeConfig(variant: ParakeetVariant) -> ParakeetModelConfig {
        ParakeetModelConfig(
            packaging: .completeMLX,
            variant: variant,
            target: "test",
            preprocessor: ParakeetPreprocessorConfig(
                sampleRate: 16_000, normalize: "per_feature", windowSize: 0.025,
                windowStride: 0.01, window: "hann", features: 4, nFFT: 512,
                dither: 0, padTo: 0, padValue: 0, preemph: 0.97
            ),
            encoder: ParakeetEncoderConfig(
                featIn: 4, layers: 1, modelDim: 4, heads: 1, ffExpansionFactor: 1,
                subsamplingFactor: 1, selfAttentionModel: "abs_pos", subsampling: "dw_striding",
                convKernelSize: 3, subsamplingConvChannels: 4, posEmbMaxLen: 16,
                causalDownsampling: false, useBias: true, xScaling: false, subsamplingConvChunkingFactor: 1
            ),
            rnntDecoder: ParakeetRNNTDecoderConfig(
                blankAsPad: true, vocabSize: 3,
                prednet: ParakeetPredictNetConfig(predHidden: 4, predRnnLayers: 1, rnnHiddenSize: nil)
            ),
            ctcDecoder: ParakeetCTCDecoderConfig(featIn: 4, numClasses: 3, vocabulary: ["▁a", "b", "."]),
            joint: variant == .ctc ? nil : makeJointConfig(),
            tdtDurations: [0, 1], maxSymbols: 4, quantizationBits: nil,
            quantizationGroupSize: nil, supportedLanguageCodes: ["en"]
        )
    }
}

private final class FixedParakeetEncoder: ParakeetExternalEncoder {
    let output: ParakeetEncoderOutput

    init(features: MLXArray, lengths: [Int]) {
        output = ParakeetEncoderOutput(features: features, lengths: lengths)
    }

    func encode(_ mel: MLXArray) throws -> ParakeetEncoderOutput { output }
}

private final class PassthroughParakeetEncoder: ParakeetExternalEncoder {
    func encode(_ mel: MLXArray) throws -> ParakeetEncoderOutput {
        ParakeetEncoderOutput(features: mel, lengths: [mel.dim(1)])
    }
}

private final class RecordingParakeetDecoder: ParakeetExternalTDTDecoder {
    let maximumBatchSize = 2
    var lengths: [Int] = []
    var callCount = 0

    func decode(encoded: MLXArray, lengths: [Int]) throws -> [ParakeetAlignedResult] {
        self.lengths = lengths
        callCount += 1
        return (0..<encoded.dim(0)).map { index in
            ParakeetAlignedResult(text: String(encoded[index, 0, 0].item(Int.self)), sentences: [])
        }
    }
}
