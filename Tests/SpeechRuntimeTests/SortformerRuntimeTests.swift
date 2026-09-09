import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import AudioSortformer

final class SortformerRuntimeTests: MLXTestCase {
    func testTinyCheckpointThroughSymlinkPreservesPredictions() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelRoot = root.appendingPathComponent("checkpoint")
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        let configData = Data(Self.tinyConfiguration.utf8)
        try configData.write(to: modelRoot.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(SortformerConfig.self, from: configData)
        let original = SortformerModel(config)
        original.train(false)
        let parameters = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        try MLX.save(arrays: parameters, url: modelRoot.appendingPathComponent("model.safetensors"))

        let link = root.appendingPathComponent("registered-model")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: modelRoot)
        let restored = try SortformerModel.fromModelDirectory(link)
        restored.train(false)
        let features = MLXArray((0..<256).map { Float($0 % 17) / 17 }).reshaped(1, 8, 32)
        let lengths = MLXArray([Int32(32)])
        let expected = original(features, audioSignalLength: lengths)
        let actual = restored(features, audioSignalLength: lengths)
        eval(expected, actual)

        XCTAssertEqual(actual.shape, [1, 4, 2])
        XCTAssertTrue(actual.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertLessThan(MLX.max(MLX.abs(expected - actual)).item(Float.self), 1e-6)
    }

    func testSegmentPostprocessingPreservesSpeakerAndTimeBoundaries() {
        let predictions = MLXArray([
            Float(0.9), 0, 0.8, 0.6, 0.1, 0.7, 0.8, 0.1, 0.9, 0
        ]).reshaped(5, 2)
        let segments = SortformerModel.predsToSegments(
            predictions, frameDuration: 0.1, threshold: 0.5, minDuration: 0.15, mergeGap: 0.11
        )
        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].speaker, 0)
        XCTAssertEqual(segments[0].start, 0, accuracy: 1e-6)
        XCTAssertEqual(segments[0].end, 0.5, accuracy: 1e-6)
        XCTAssertEqual(segments[1].speaker, 1)
        XCTAssertEqual(segments[1].start, 0.1, accuracy: 1e-6)
        XCTAssertEqual(segments[1].end, 0.3, accuracy: 1e-6)
    }

    private static let tinyConfiguration = """
    {
      "num_speakers": 2,
      "fc_encoder_config": {
        "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "intermediate_size": 16, "num_mel_bins": 8,
        "conv_kernel_size": 3, "subsampling_conv_channels": 2
      },
      "tf_encoder_config": {
        "d_model": 8, "encoder_layers": 1, "encoder_attention_heads": 2,
        "encoder_ffn_dim": 16
      },
      "modules_config": {"num_speakers": 2, "fc_d_model": 8, "tf_d_model": 8},
      "processor_config": {"feature_size": 8}
    }
    """

    func testX86LinuxRuntimePromotionOnlyConvertsFloat16Weights() {
        let weights = [
            "half": MLX.ones([2], dtype: .float16),
            "float": MLX.ones([2], dtype: .float32),
            "integer": MLX.ones([2], dtype: .int32),
        ]

        let promoted = SortformerModel.runtimeCompatibleWeights(weights, promoteFloat16: true)

        XCTAssertEqual(promoted["half"]?.dtype, .float32)
        XCTAssertEqual(promoted["float"]?.dtype, .float32)
        XCTAssertEqual(promoted["integer"]?.dtype, .int32)
    }

    func testRTTMRendersStableAnonymousSpeakerLabels() {
        let output = DiarizationOutput(
            segments: [
                DiarizationSegment(start: 0, end: 1.25, speaker: 0),
                DiarizationSegment(start: 2.5, end: 4, speaker: 2),
            ],
            numSpeakers: 2,
            totalTime: 0.1
        )

        XCTAssertEqual(
            output.rttm(fileID: "meeting"),
            """
            SPEAKER meeting 1 0.000 1.250 <NA> <NA> speaker_0 <NA> <NA>
            SPEAKER meeting 1 2.500 1.500 <NA> <NA> speaker_2 <NA> <NA>
            """
        )
    }

}
