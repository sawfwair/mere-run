import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class FourDAnyoneContractTests: MereRunCoreTestCase {
    func testCameraOrdersAndRoutingMatchUpstream() throws {
        let manifest = try FourDAnyoneFixture.manifest()
        XCTAssertEqual(manifest.upstreamRevision, FourDAnyoneModelLoader.upstreamRevision)
        XCTAssertEqual(manifest.modelRevision, FourDAnyoneModelLoader.modelRevision)
        for fixture in manifest.plans {
            let plan = try FourDAnyoneViewPlan(
                viewsPerLayer: fixture.viewsPerLayer, layerPitches: fixture.pitches,
                targetContextRouting: fixture.routing
            )
            XCTAssertEqual(plan.cameraOrder, fixture.order)
            for (step, expected) in fixture.routes.enumerated() {
                XCTAssertEqual(plan.groups(step: step), expected)
                XCTAssertEqual(plan.groups(step: step).flatMap { $0 }.sorted(), Array(0..<plan.viewCount))
            }
        }
        XCTAssertFalse(try FourDAnyoneViewPlan().referencePacking)
        XCTAssertThrowsError(try FourDAnyoneViewPlan(viewsPerLayer: 5))
        XCTAssertThrowsError(try FourDAnyoneViewPlan(layerPitches: [15, 15]))
        XCTAssertThrowsError(try FourDAnyoneViewPlan(layerPitches: [46]))
        XCTAssertThrowsError(try FourDAnyoneViewPlan(viewsPerLayer: Int.max, layerPitches: [0, 15]))
    }

    func testScheduleMatchesUpstreamIncludingTerminalStep() throws {
        let tensors = try FourDAnyoneFixture.tensors()
        for count in [4, 24] {
            let schedule = try FourDAnyoneSchedule(steps: count)
            FourDAnyoneFixture.assertClose(MLXArray(schedule.sigmas), tensors["schedule\(count)"]!, tolerance: 1e-7)
            XCTAssertEqual(schedule.timesteps.count, count)
            XCTAssertEqual(schedule.sigmas.last, 0)
        }
        XCTAssertThrowsError(try FourDAnyoneSchedule(steps: 0))
        let schedule = try FourDAnyoneSchedule(steps: 4)
        let result = schedule.step(
            prediction: MLXArray([Float(2)]).asType(.bfloat16),
            sample: MLXArray([Float(1)]).asType(.bfloat16), index: 3
        )
        XCTAssertEqual(result.dtype, .bfloat16)
        XCTAssertEqual(result.item(Float.self), -0.25)
    }

    func testProductionSchemaMatchesAll1182ReleasedTensorShapes() throws {
        let schema = try JSONDecoder().decode(
            [String: [Int]].self, from: Data(contentsOf: FourDAnyoneFixture.url("checkpoint-schema", extension: "json"))
        )
        let transformer = FourDAnyoneTransformerModel()
        let pose = FourDAnyonePoseEncoder()
        let native = FourDAnyoneModelLoader.transformerSchema(transformer)
            .merging(FourDAnyoneModelLoader.poseSchema(pose)) { first, _ in first }
        XCTAssertEqual(native.count, 1_182)
        XCTAssertEqual(native, schema)
        let metadata = schema.mapValues {
            SafetensorsStreamingLoader.TensorMetadata(shape: $0, dtype: .bfloat16, startOffset: 0, endOffset: 0)
        }
        try FourDAnyoneModelLoader.validateCheckpoint(
            metadata: metadata, transformer: transformer, pose: pose, partition: .transformer
        )
        var missing = metadata
        missing.removeValue(forKey: "blocks.0.self_attn_mvs.q.weight")
        XCTAssertThrowsError(try FourDAnyoneModelLoader.validateCheckpoint(
            metadata: missing, transformer: transformer, pose: pose, partition: .transformer
        ))
        var unknown = metadata
        unknown["unknown.weight"] = metadata.values.first!
        XCTAssertThrowsError(try FourDAnyoneModelLoader.validateCheckpoint(
            metadata: unknown, transformer: transformer, pose: pose, partition: .pose
        ))
        var malformed = metadata
        malformed["pose_encoder.scale"] = .init(shape: [2], dtype: .bfloat16, startOffset: 0, endOffset: 0)
        XCTAssertThrowsError(try FourDAnyoneModelLoader.validateCheckpoint(
            metadata: malformed, transformer: transformer, pose: pose, partition: .pose
        ))
        let onlyTransformer = metadata.filter { !$0.key.hasPrefix("pose_encoder.") }
        try FourDAnyoneModelLoader.validateCheckpoint(
            metadata: onlyTransformer, transformer: transformer, pose: pose, partition: .transformer
        )
        var partialCompanion = onlyTransformer
        partialCompanion["pose_encoder.scale"] = metadata["pose_encoder.scale"]
        XCTAssertThrowsError(try FourDAnyoneModelLoader.validateCheckpoint(
            metadata: partialCompanion, transformer: transformer, pose: pose, partition: .transformer
        ))
    }

    func testPromptContextRequiresReleasedMetadataShapeAndDtype() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("4danyone-prompt-\(UUID()).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let context = MLX.zeros([1, 512, 4_096], dtype: .bfloat16)
        var metadata = [
            "format": "fdanyone.prompt_context", "version": "2", "prompt": "视频中的人在做动作",
            "source_repo": "Wan-AI/Wan2.1-T2V-1.3B",
            "source_revision": "3f40b6dc4ca5c02dd23c9db74d9d2ccb82903b86",
            "text_encoder_sha256": String(repeating: "a", count: 64),
            "tokenizer_manifest_sha256": String(repeating: "b", count: 64),
        ]
        // Synthetic contents test the upstream metadata contract, not trained embeddings.
        try MLX.save(arrays: ["context": context], metadata: metadata, url: url)
        XCTAssertEqual(try FourDAnyoneModelLoader.loadPromptContext(from: url).shape, context.shape)
        metadata["version"] = "1"
        try MLX.save(arrays: ["context": context], metadata: metadata, url: url)
        XCTAssertThrowsError(try FourDAnyoneModelLoader.loadPromptContext(from: url))
        metadata["version"] = "2"
        try MLX.save(arrays: ["context": context.asType(.float32)], metadata: metadata, url: url)
        XCTAssertThrowsError(try FourDAnyoneModelLoader.loadPromptContext(from: url))
        try MLX.save(arrays: ["context": context[0..., 0..<1, 0...]], metadata: metadata, url: url)
        XCTAssertThrowsError(try FourDAnyoneModelLoader.loadPromptContext(from: url))
    }
}
