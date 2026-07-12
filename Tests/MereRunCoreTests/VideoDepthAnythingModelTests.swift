import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class VideoDepthAnythingModelTests: MereRunCoreTestCase {
    func testDINOv2UsesPyTorchOffsetSamplingGrid() {
        let coordinates = dinoV2PyTorchInterpolationCoordinates(
            sourceSize: 37,
            outputSize: 4,
            offset: 0.1
        )
        MLX.eval(coordinates)
        let expected: [Float] = [
            4.012_195,
            13.036_585,
            22.060_976,
            31.085_365,
        ]
        for (actual, expected) in zip(coordinates.asArray(Float.self), expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
    }

    func testDefaultGraphMatchesPinnedSmallCheckpointInventory() {
        let model = VideoDepthAnythingModel()
        let parameters = model.parameters().flattened()
        XCTAssertEqual(parameters.count, VideoDepthAnythingWeights.sourceTensorCount)
        XCTAssertEqual(
            parameters.reduce(0) { $0 + $1.1.shape.reduce(1, *) },
            VideoDepthAnythingWeights.inferenceScalarCount
        )

        let shapes = Dictionary(uniqueKeysWithValues: parameters.map { ($0.0, $0.1.shape) })
        XCTAssertEqual(shapes["pretrained.cls_token"], [1, 1, 384])
        XCTAssertEqual(shapes["pretrained.class_position"], [1, 1, 384])
        XCTAssertEqual(shapes["pretrained.patch_position"], [1, 1_369, 384])
        XCTAssertEqual(shapes["pretrained.blocks.11.attn.qkv.weight"], [1_152, 384])
        XCTAssertEqual(shapes["head.projects.2.weight"], [192, 1, 1, 384])
        XCTAssertEqual(shapes["head.resize_layers.0.weight"], [48, 4, 4, 48])
        XCTAssertEqual(shapes["head.resize_layers.3.weight"], [384, 3, 3, 384])
        XCTAssertEqual(
            shapes["head.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.to_q.weight"],
            [192, 192]
        )
        let firstAttention = "head.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0"
        XCTAssertEqual(shapes["\(firstAttention).to_out.0.weight"], [192, 192])
        XCTAssertEqual(
            shapes["head.motion_modules.0.temporal_transformer.transformer_blocks.0.ff.net.0.proj.weight"],
            [1_536, 192]
        )
        XCTAssertEqual(
            shapes["head.motion_modules.0.temporal_transformer.transformer_blocks.0.ff.net.2.weight"],
            [192, 768]
        )
    }

    func testMiniatureGraphRunsTemporalDPTForward() {
        let backbone = DINOv2Configuration(
            hiddenSize: 12,
            layerCount: 4,
            headCount: 3,
            intermediateSize: 24,
            patchSize: 2,
            positionGridSize: 2
        )
        let configuration = VideoDepthAnythingConfiguration(
            backbone: backbone,
            intermediateLayers: [0, 1, 2, 3],
            featureChannels: 8,
            projectedChannels: [4, 8, 16, 32],
            temporalFrameCount: 4,
            temporalHeadCount: 4,
            temporalTransformerBlockCount: 1,
            temporalAttentionBlockCount: 2,
            temporalGroupCount: 4
        )
        let model = VideoDepthAnythingModel(configuration: configuration)
        let input = MLX.zeros([1, 3, 8, 8, 3], dtype: .float32)
        let output = model(input).depth
        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 3, 8, 8])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy { $0.isFinite && $0 >= 0 })
    }

    func testEncoderAndDPTMicroBatchingMatchFullBatch() {
        let backbone = DINOv2Configuration(
            hiddenSize: 12,
            layerCount: 4,
            headCount: 3,
            intermediateSize: 24,
            patchSize: 2,
            positionGridSize: 2
        )
        let configuration = VideoDepthAnythingConfiguration(
            backbone: backbone,
            intermediateLayers: [0, 1, 2, 3],
            featureChannels: 8,
            projectedChannels: [4, 8, 16, 32],
            temporalFrameCount: 4,
            temporalHeadCount: 4,
            temporalTransformerBlockCount: 1,
            temporalAttentionBlockCount: 2,
            temporalGroupCount: 4
        )
        let model = VideoDepthAnythingModel(configuration: configuration)
        XCTAssertEqual(model.defaultMemoryConfiguration.encoderMicroBatchSize, 4)
        XCTAssertEqual(model.defaultMemoryConfiguration.dptTailMicroBatchSize, 4)

        let elementCount = 1 * 4 * 8 * 8 * 3
        let input = (MLX.arange(elementCount, dtype: .float32) / Float(elementCount) - 0.5)
            .reshaped(1, 4, 8, 8, 3)
        let fullBatch = model(input, memoryConfiguration: .fullBatch).depth
        MLX.eval(fullBatch)
        let microBatched = model(
            input,
            memoryConfiguration: VideoDepthAnythingMemoryConfiguration(
                encoderMicroBatchSize: 2,
                dptTailMicroBatchSize: 2
            )
        ).depth
        MLX.eval(microBatched)

        let differences = zip(
            fullBatch.asArray(Float.self),
            microBatched.asArray(Float.self)
        ).map { abs($0 - $1) }
        let mean = differences.reduce(0, +) / Float(differences.count)
        let maximum = differences.max() ?? 0
        XCTAssertLessThanOrEqual(mean, 1e-6, "micro-batch MAE \(mean), max \(maximum)")
        XCTAssertLessThanOrEqual(maximum, 1e-5, "micro-batch MAE \(mean), max \(maximum)")
    }

    func testStrictCheckpointLoadAndNativeForwardWhenFixtureIsAvailable() throws {
        guard let fixturePath = ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_SAFETENSORS"],
              FileManager.default.fileExists(atPath: fixturePath)
        else {
            throw XCTSkip("Set MERERUN_TEST_VDA_SAFETENSORS to run the full VDA-S checkpoint test")
        }

        let model = VideoDepthAnythingModel()
        try VideoDepthAnythingWeights.load(
            model: model,
            safetensorsURL: URL(fileURLWithPath: fixturePath)
        )
        try assertNativeForward(
            model: model,
            parityRoot: ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_PARITY"],
            label: "safetensors"
        )
    }

    func testStrictPTHLoadAndNativeForwardWhenFixtureIsAvailable() throws {
        guard let fixturePath = ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_PTH"],
              FileManager.default.fileExists(atPath: fixturePath)
        else {
            throw XCTSkip("Set MERERUN_TEST_VDA_PTH to run the official VDA-S checkpoint test")
        }

        let archive = try PyTorchStateDictArchive(url: URL(fileURLWithPath: fixturePath))
        let model = VideoDepthAnythingModel()
        try VideoDepthAnythingWeights.load(model: model, archive: archive)
        try assertNativeForward(
            model: model,
            parityRoot: ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_PARITY"],
            label: "pth"
        )
    }

    func testStrictMetricPTHLoadAndNativeForwardWhenFixtureIsAvailable() throws {
        guard let fixturePath = ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_METRIC_PTH"],
              FileManager.default.fileExists(atPath: fixturePath)
        else {
            throw XCTSkip("Set MERERUN_TEST_VDA_METRIC_PTH to run the official metric VDA-S checkpoint test")
        }

        let archive = try PyTorchStateDictArchive(url: URL(fileURLWithPath: fixturePath))
        let model = VideoDepthAnythingModel()
        try VideoDepthAnythingWeights.load(model: model, archive: archive)
        try assertNativeForward(
            model: model,
            parityRoot: ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_METRIC_PARITY"],
            label: "metric pth"
        )
    }

    func testSourceWeightMapperSplitsPositionAndDistinguishesTransposeConvolution() {
        let position = MLX.zeros([1, 5, 4], dtype: .float32)
        let mappedPosition = VideoDepthAnythingWeights.mapSourceTensor(
            key: "pretrained.pos_embed",
            value: position
        )
        XCTAssertEqual(mappedPosition.map(\.0), ["pretrained.class_position", "pretrained.patch_position"])
        XCTAssertEqual(mappedPosition.map { $0.1.shape }, [[1, 1, 4], [1, 4, 4]])

        let transposed = VideoDepthAnythingWeights.mapSourceTensor(
            key: "head.resize_layers.0.weight",
            value: MLX.zeros([4, 6, 2, 2], dtype: .float32)
        )
        XCTAssertEqual(transposed.single?.1.shape, [6, 2, 2, 4])

        let convolution = VideoDepthAnythingWeights.mapSourceTensor(
            key: "head.projects.0.weight",
            value: MLX.zeros([6, 4, 2, 2], dtype: .float32)
        )
        XCTAssertEqual(convolution.single?.1.shape, [6, 2, 2, 4])
        XCTAssertTrue(
            VideoDepthAnythingWeights.mapSourceTensor(
                key: "pretrained.mask_token",
                value: MLX.zeros([1, 4])
            ).isEmpty
        )
    }

    private func assertNativeForward(
        model: VideoDepthAnythingModel,
        parityRoot: String?,
        label: String
    ) throws {
        let input: MLXArray
        if let parityRoot, !parityRoot.isEmpty {
            let values = try readFloat32(
                URL(fileURLWithPath: parityRoot).appendingPathComponent("input.f32")
            )
            XCTAssertEqual(values.count, 1 * 4 * 56 * 56 * 3)
            input = MLXArray(values).reshaped(1, 4, 56, 56, 3)
        } else {
            input = MLX.zeros([1, 2, 56, 56, 3], dtype: .float32)
        }
        let output = model(
            input,
            memoryConfiguration: VideoDepthAnythingMemoryConfiguration(
                encoderMicroBatchSize: 2,
                dptTailMicroBatchSize: 2
            )
        ).depth
        MLX.eval(output)

        XCTAssertEqual(output.shape, [1, input.dim(1), 56, 56])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy { $0.isFinite && $0 >= 0 })

        if let parityRoot, !parityRoot.isEmpty {
            let expected = try readFloat32(
                URL(fileURLWithPath: parityRoot).appendingPathComponent("depth.f32")
            )
            let actual = output.asArray(Float.self)
            XCTAssertEqual(actual.count, expected.count)
            guard actual.count == expected.count else { return }
            let differences = zip(actual, expected).map { abs($0 - $1) }
            let mean = differences.reduce(0, +) / Float(Swift.max(1, differences.count))
            let maximum = differences.max() ?? 0
            print("VDA \(label) native parity: MAE=\(mean), max=\(maximum)")
            XCTAssertLessThanOrEqual(mean, 5e-4, "VDA depth MAE \(mean), max \(maximum)")
            XCTAssertLessThanOrEqual(maximum, 5e-3, "VDA depth MAE \(mean), max \(maximum)")
        }
    }

    private func readFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        precondition(data.count.isMultiple(of: 4))
        return stride(from: 0, to: data.count, by: 4).map { offset in
            let bits = UInt32(data[offset])
                | (UInt32(data[offset + 1]) << 8)
                | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
            return Float(bitPattern: bits)
        }
    }
}

private extension Array {
    var single: Element? {
        count == 1 ? first : nil
    }
}
