import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class DepthAnything3ModelTests: MereRunCoreTestCase {
    func testPinnedSmallCheckpointProvenanceAndInventory() {
        XCTAssertEqual(DepthAnything3SmallCheckpoint.repository, "depth-anything/DA3-SMALL")
        XCTAssertEqual(DepthAnything3SmallCheckpoint.revision, "e08cab65ca0ec38e7826075418411ab90cab4da3")
        XCTAssertEqual(DepthAnything3SmallCheckpoint.license, "Apache-2.0")
        XCTAssertEqual(DepthAnything3SmallCheckpoint.artifact.byteCount, 137_248_940)
        XCTAssertEqual(
            DepthAnything3SmallCheckpoint.artifact.sha256,
            "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
        )
        XCTAssertEqual(DepthAnything3SmallCheckpoint.configurationArtifact.byteCount, 1_202)
        XCTAssertEqual(
            DepthAnything3SmallCheckpoint.configurationArtifact.sha256,
            "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2"
        )

        let model = DepthAnything3Model()
        let parameters = model.parameters().flattened()
        XCTAssertEqual(parameters.count, DepthAnything3SmallCheckpoint.tensorCount)
        XCTAssertEqual(
            parameters.reduce(0) { $0 + $1.1.shape.reduce(1, *) },
            DepthAnything3SmallCheckpoint.scalarCount
        )
        let shapes = Dictionary(uniqueKeysWithValues: parameters.map { ($0.0, $0.1.shape) })
        XCTAssertEqual(shapes["backbone.pretrained.pos_embed"], [1, 1_370, 384])
        XCTAssertEqual(shapes["backbone.pretrained.camera_token"], [1, 2, 384])
        XCTAssertEqual(shapes["backbone.pretrained.blocks.4.attn.q_norm.weight"], [64])
        XCTAssertNil(shapes["backbone.pretrained.blocks.3.attn.q_norm.weight"])
        XCTAssertEqual(shapes["head.norm.weight"], [768])
        XCTAssertEqual(shapes["head.projects.0.weight"], [48, 1, 1, 768])
        XCTAssertEqual(shapes["head.resize_layers.0.weight"], [48, 4, 4, 48])
        XCTAssertEqual(shapes["head.scratch.output_conv2_aux.0.owned_shared_norm.weight"], [32])
        XCTAssertNil(shapes["head.scratch.output_conv2_aux.3.owned_shared_norm.weight"])
        XCTAssertEqual(shapes["cam_dec.backbone.first.weight"], [768, 768])
        XCTAssertEqual(shapes["cam_enc.pose_branch.fc1.weight"], [192, 9])
    }

    func testSourceMapperRemovesRootAndDistinguishesTransposeConvolution() {
        let convolution = DepthAnything3Weights.mapSourceTensor(
            key: "model.head.projects.0.weight",
            value: MLX.zeros([6, 4, 2, 3], dtype: .float32)
        )
        XCTAssertEqual(convolution.map(\.0), ["head.projects.0.weight"])
        XCTAssertEqual(convolution.first?.1.shape, [6, 2, 3, 4])

        let transposed = DepthAnything3Weights.mapSourceTensor(
            key: "model.head.resize_layers.0.weight",
            value: MLX.zeros([4, 6, 2, 3], dtype: .float32)
        )
        XCTAssertEqual(transposed.first?.1.shape, [6, 2, 3, 4])
        XCTAssertTrue(
            DepthAnything3Weights.mapSourceTensor(
                key: "unexpected.weight",
                value: MLX.zeros([1])
            ).isEmpty
        )
    }

    func testUVPositionEmbeddingMatchesAuthoritativeGridEndpoints() {
        let embedding = depthAnything3UVPositionEmbedding(
            height: 2,
            width: 3,
            channels: 4,
            aspectRatio: 1.5
        )
        MLX.eval(embedding)
        XCTAssertEqual(embedding.shape, [1, 2, 3, 4])
        let values = embedding.asArray(Float.self)
        let spanX = Float(1.5 / sqrt(1.5 * 1.5 + 1))
        let spanY = Float(1 / sqrt(1.5 * 1.5 + 1))
        let left = -spanX * 2 / 3
        let top = -spanY / 2
        XCTAssertEqual(values[0], sin(left), accuracy: 1e-6)
        XCTAssertEqual(values[1], cos(left), accuracy: 1e-6)
        XCTAssertEqual(values[2], sin(top), accuracy: 1e-6)
        XCTAssertEqual(values[3], cos(top), accuracy: 1e-6)
    }

    func testMiniatureGraphRunsUnconditionedAndPoseConditioned() throws {
        let model = DepthAnything3Model(configuration: try miniatureConfiguration())
        let input = (MLX.arange(1 * 2 * 8 * 8 * 3, dtype: .float32) / 384 - 0.5)
            .reshaped(1, 2, 8, 8, 3)
        let unconditioned = model(input, referenceViewStrategy: .first)
        MLX.eval(
            unconditioned.depth,
            unconditioned.confidence,
            unconditioned.extrinsics,
            unconditioned.intrinsics,
            unconditioned.ray,
            unconditioned.rayConfidence
        )
        assertOutputShapes(unconditioned, views: 2, height: 8, width: 8)

        let extrinsics = MLXArray([
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
            1, 0, 0, -0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        ]).reshaped(1, 2, 4, 4)
        let intrinsics = MLXArray([
            8, 0, 4, 0, 8, 4, 0, 0, 1,
            8, 0, 4, 0, 8, 4, 0, 0, 1,
        ]).reshaped(1, 2, 3, 3)
        let conditioned = model(
            input,
            cameraConditioning: try DepthAnything3CameraConditioning(
                extrinsics: extrinsics,
                intrinsics: intrinsics
            ),
            referenceViewStrategy: .first
        )
        MLX.eval(conditioned.depth, conditioned.extrinsics, conditioned.intrinsics)
        assertOutputShapes(conditioned, views: 2, height: 8, width: 8)
    }

    func testPinnedCheckpointStrictLoadAndParityWhenFixturesAreAvailable() throws {
        guard let checkpointPath = ProcessInfo.processInfo.environment["MERERUN_TEST_DA3_SAFETENSORS"],
              FileManager.default.fileExists(atPath: checkpointPath)
        else {
            throw XCTSkip("Set MERERUN_TEST_DA3_SAFETENSORS to run DA3-Small strict loading")
        }
        let model = DepthAnything3Model()
        try DepthAnything3Weights.load(
            model: model,
            safetensorsURL: URL(fileURLWithPath: checkpointPath)
        )

        guard let parityRoot = ProcessInfo.processInfo.environment["MERERUN_TEST_DA3_PARITY"],
              !parityRoot.isEmpty
        else { return }
        let root = URL(fileURLWithPath: parityRoot)
        let provenance = try JSONDecoder().decode(
            ParityProvenance.self,
            from: Data(contentsOf: root.appendingPathComponent("provenance.json"))
        )
        XCTAssertEqual(provenance.checkpointRepository, DepthAnything3SmallCheckpoint.repository)
        XCTAssertEqual(provenance.checkpointRevision, DepthAnything3SmallCheckpoint.revision)
        XCTAssertEqual(provenance.checkpointSHA256, DepthAnything3SmallCheckpoint.artifact.sha256)
        XCTAssertEqual(
            provenance.configSHA256,
            DepthAnything3SmallCheckpoint.configurationArtifact.sha256
        )
        XCTAssertEqual(
            provenance.sourceRepository,
            DepthAnything3SmallCheckpoint.upstreamSourceRepository
        )
        XCTAssertEqual(provenance.sourceRevision, DepthAnything3SmallCheckpoint.upstreamSourceRevision)
        XCTAssertEqual(provenance.license, DepthAnything3SmallCheckpoint.license)
        let inputValues = try readFloat32(root.appendingPathComponent("input.f32"))
        let shape = try JSONDecoder().decode(
            ParityShape.self,
            from: Data(contentsOf: root.appendingPathComponent("shape.json"))
        )
        let input = MLXArray(inputValues).reshaped(1, shape.views, shape.height, shape.width, 3)
        let output = model(input, referenceViewStrategy: .first)
        MLX.eval(output.depth, output.confidence, output.extrinsics, output.intrinsics)
        try assertParity(output.depth, file: "depth.f32", root: root, mean: 1e-5, maximum: 1e-4)
        try assertParity(output.confidence, file: "confidence.f32", root: root, mean: 1e-5, maximum: 1e-4)
        try assertParity(output.extrinsics, file: "extrinsics.f32", root: root, mean: 1e-5, maximum: 1e-4)
        try assertParity(output.intrinsics, file: "intrinsics.f32", root: root, mean: 1e-4, maximum: 1e-3)

        let conditioningExtrinsics = MLXArray(
            try readFloat32(root.appendingPathComponent("conditioning-extrinsics.f32"))
        ).reshaped(1, shape.views, 4, 4)
        let conditioningIntrinsics = MLXArray(
            try readFloat32(root.appendingPathComponent("conditioning-intrinsics.f32"))
        ).reshaped(1, shape.views, 3, 3)
        let conditioned = model(
            input,
            cameraConditioning: try DepthAnything3CameraConditioning(
                extrinsics: conditioningExtrinsics,
                intrinsics: conditioningIntrinsics
            ),
            referenceViewStrategy: .first
        )
        MLX.eval(
            conditioned.depth,
            conditioned.confidence,
            conditioned.extrinsics,
            conditioned.intrinsics
        )
        try assertParity(conditioned.depth, file: "conditioned-depth.f32", root: root, mean: 1e-5, maximum: 1e-4)
        try assertParity(conditioned.confidence, file: "conditioned-confidence.f32", root: root, mean: 1e-5, maximum: 1e-4)
        try assertParity(conditioned.extrinsics, file: "conditioned-extrinsics.f32", root: root, mean: 1e-5, maximum: 1e-4)
        try assertParity(conditioned.intrinsics, file: "conditioned-intrinsics.f32", root: root, mean: 1e-4, maximum: 1e-3)
    }

    private func miniatureConfiguration() throws -> DepthAnything3Configuration {
        try DepthAnything3Configuration(
            hiddenSize: 32,
            layerCount: 8,
            headCount: 4,
            intermediateSize: 64,
            patchSize: 2,
            positionGridSize: 2,
            outputLayers: [4, 5, 6, 7],
            alternateAttentionStart: 2,
            queryKeyNormStart: 2,
            rotaryEmbeddingStart: 2,
            featureChannels: 8,
            projectedChannels: [4, 8, 16, 32],
            cameraEncoderDepth: 2,
            cameraEncoderHeadCount: 4,
            headMicroBatchSize: nil
        )
    }

    private func assertOutputShapes(
        _ output: DepthAnything3RawOutput,
        views: Int,
        height: Int,
        width: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(output.depth.shape, [1, views, height, width], file: file, line: line)
        XCTAssertEqual(output.confidence.shape, [1, views, height, width], file: file, line: line)
        XCTAssertEqual(output.extrinsics.shape, [1, views, 3, 4], file: file, line: line)
        XCTAssertEqual(output.intrinsics.shape, [1, views, 3, 3], file: file, line: line)
        // The authoritative auxiliary ray pyramid is emitted at 8x the patch
        // grid and is intentionally not resized to the primary depth size.
        let rayHeight = height / 2 * 8
        let rayWidth = width / 2 * 8
        XCTAssertEqual(output.ray.shape, [1, views, rayHeight, rayWidth, 6], file: file, line: line)
        XCTAssertEqual(output.rayConfidence.shape, [1, views, rayHeight, rayWidth], file: file, line: line)
        XCTAssertTrue(output.depth.asArray(Float.self).allSatisfy { $0.isFinite && $0 > 0 }, file: file, line: line)
        XCTAssertTrue(output.confidence.asArray(Float.self).allSatisfy { $0.isFinite && $0 >= 1 }, file: file, line: line)
    }

    private func assertParity(
        _ actualArray: MLXArray,
        file: String,
        root: URL,
        mean thresholdMean: Float,
        maximum thresholdMaximum: Float
    ) throws {
        let expected = try readFloat32(root.appendingPathComponent(file))
        let actual = actualArray.asArray(Float.self)
        XCTAssertEqual(actual.count, expected.count, file)
        guard actual.count == expected.count else { return }
        let differences = zip(actual, expected).map { abs($0 - $1) }
        let mean = differences.reduce(0, +) / Float(max(1, differences.count))
        let maximum = differences.max() ?? 0
        print("DA3 \(file) parity: MAE=\(mean), max=\(maximum)")
        XCTAssertLessThanOrEqual(mean, thresholdMean, "\(file) MAE \(mean), max \(maximum)")
        XCTAssertLessThanOrEqual(maximum, thresholdMaximum, "\(file) MAE \(mean), max \(maximum)")
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

private struct ParityShape: Decodable {
    let views: Int
    let height: Int
    let width: Int
}

private struct ParityProvenance: Decodable {
    let checkpointRepository: String
    let checkpointRevision: String
    let checkpointSHA256: String
    let configSHA256: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String

    private enum CodingKeys: String, CodingKey {
        case checkpointRepository = "checkpoint_repository"
        case checkpointRevision = "checkpoint_revision"
        case checkpointSHA256 = "checkpoint_sha256"
        case configSHA256 = "config_sha256"
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case license
    }
}
