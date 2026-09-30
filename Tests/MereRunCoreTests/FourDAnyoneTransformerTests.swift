import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class FourDAnyoneTransformerTests: MereRunCoreTestCase {
    func testDirectAndPackedPredictionsMatchUpstream() throws {
        let tensors = try FourDAnyoneFixture.tensors()
        let model = try FourDAnyoneFixture.model()
        for packed in [false, true] {
            let prefix = packed ? "packed" : "direct"
            let input = FourDAnyoneFixture.input(tensors, packed: packed)
            let grid = try input.validate(configuration: model.configuration)
            let assembled = model.assemble(input, grid: grid)
            FourDAnyoneFixture.assertClose(assembled, tensors[prefix + ".assembly"]!)
            let time = model.embeddedTime(timestep: 625, targets: 4, packed: packed ? 2 : 1)
            FourDAnyoneFixture.assertClose(time, tensors[prefix + ".time"]!)
            let firstBlock = model.blocks[0](
                assembled, context: model.textEmbedding(input.promptContext),
                time: model.timeProjection(time).reshaped(-1, 6, 24), grid: grid,
                spatialRoPE: FourDAnyoneRoPE.prepare(grid: grid, headDimension: 12),
                multiviewRoPE: FourDAnyoneRoPE.prepare(
                    grid: Wan2GridSize(frames: packed ? 6 : 5, height: 3, width: 5), headDimension: 12
                ), maximumQueryTokens: 11
            )
            FourDAnyoneFixture.assertClose(firstBlock, tensors[prefix + ".block0"]!)
            let prediction = try model(input, maximumQueryTokens: 11)
            FourDAnyoneFixture.assertClose(prediction, tensors[prefix + ".prediction"]!)
            FourDAnyoneFixture.assertClose(try model(input, maximumQueryTokens: 512), prediction)
        }
    }

    func testFourStepRoutedGenerationMatchesUpstream() throws {
        let tensors = try FourDAnyoneFixture.tensors()
        let generator = FourDAnyoneGenerator(transformer: try FourDAnyoneFixture.model())
        let plan = try FourDAnyoneViewPlan(viewsPerLayer: 8)
        let conditioning = FourDAnyonePreparedConditioning(
            sources: tensors["sources"]!, poseFeatures: tensors["generation.poses"]!,
            nullPoseFeatures: tensors["null"]!, promptContext: tensors["context"]!
        )
        let initial = tensors["generation.initial"]!
        let before = initial.asArray(Float.self)
        var events: [FourDAnyoneProgress] = []
        let output = try generator.generate(
            initialLatents: initial, conditioning: conditioning, plan: plan, steps: 4,
            maximumQueryTokens: 13, progress: { events.append($0) }
        )
        FourDAnyoneFixture.assertClose(output, tensors["generation.step4"]!, tolerance: 5e-5)
        XCTAssertEqual(initial.asArray(Float.self), before)
        XCTAssertEqual(events.count, 8)
        for (index, event) in events.enumerated() {
            XCTAssertEqual(event.step, index / 2 + 1)
            XCTAssertEqual(event.cameraIDs, plan.groups(step: index / 2)[index % 2])
            XCTAssertEqual(event.completedGroups, index % 2 + 1)
        }
    }

    func testBF16StoragePreservesFP32RoutedGeneration() throws {
        let tensors = try FourDAnyoneFixture.tensors()
        let stored = try FourDAnyoneModelLoader.loadTransformer(
            from: FourDAnyoneFixture.url("transformer", extension: "safetensors"),
            configuration: FourDAnyoneFixture.configuration
        )
        let reference = try FourDAnyoneFixture.model()
        try reference.update(parameters: ModuleParameters.unflattened(
            stored.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        ), verify: .noUnusedKeys)
        let conditioning = FourDAnyonePreparedConditioning(
            sources: tensors["sources"]!, poseFeatures: tensors["generation.poses"]!,
            nullPoseFeatures: tensors["null"]!, promptContext: tensors["context"]!
        )
        let plan = try FourDAnyoneViewPlan(viewsPerLayer: 8)
        let initial = tensors["generation.initial"]!
        let actual = try FourDAnyoneGenerator(transformer: stored).generate(
            initialLatents: initial, conditioning: conditioning, plan: plan, steps: 4
        )
        let expected = try FourDAnyoneGenerator(transformer: reference).generate(
            initialLatents: initial, conditioning: conditioning, plan: plan, steps: 4
        )
        XCTAssertEqual(actual.dtype, .float32)
        XCTAssertTrue(stored.parameters().flattened().allSatisfy { $0.1.dtype == .bfloat16 })
        FourDAnyoneFixture.assertClose(actual, expected, tolerance: 1e-6)
    }

    func testConditioningAffectsTargetsAndSourceTokens() throws {
        let tensors = try FourDAnyoneFixture.tensors()
        let model = try FourDAnyoneFixture.model()
        let baseline = try model(FourDAnyoneFixture.input(tensors, packed: true))
        var changed = tensors
        changed["sources"] = tensors["sources"]! + 0.25
        let updated = try model(FourDAnyoneFixture.input(changed, packed: true))
        XCTAssertGreaterThan(MLX.max(MLX.abs(updated - baseline)).item(Float.self), 1e-4)
        changed = tensors
        changed["null"] = MLX.zeros(tensors["null"]!.shape)
        let withoutLearnedNull = try model(FourDAnyoneFixture.input(changed, packed: true))
        XCTAssertGreaterThan(MLX.max(MLX.abs(withoutLearnedNull - baseline)).item(Float.self), 1e-4)
    }

    func testRejectsMismatchedPreparedInputsAndHonorsCancellation() throws {
        let tensors = try FourDAnyoneFixture.tensors()
        let model = try FourDAnyoneFixture.model()
        var invalid = tensors
        invalid["poses"] = MLX.zeros([4, 24, 2, 3, 4])
        XCTAssertThrowsError(try model(FourDAnyoneFixture.input(invalid, packed: true)))
        XCTAssertThrowsError(try model(FourDAnyoneFixture.input(tensors, packed: true), maximumQueryTokens: 0))
        var checks = 0
        XCTAssertThrowsError(try model(FourDAnyoneFixture.input(tensors, packed: true), checkCancellation: {
            checks += 1
            if checks == 2 { throw CancellationError() }
        })) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(checks, 2)
        let generator = FourDAnyoneGenerator(transformer: model)
        let conditioning = FourDAnyonePreparedConditioning(
            sources: tensors["sources"]!, poseFeatures: tensors["generation.poses"]!,
            nullPoseFeatures: tensors["null"]!, promptContext: tensors["context"]!
        )
        var reports = 0
        XCTAssertThrowsError(try generator.generate(
            initialLatents: tensors["generation.initial"]!, conditioning: conditioning,
            plan: FourDAnyoneViewPlan(viewsPerLayer: 8), steps: 4,
            checkCancellation: { if reports == 1 { throw CancellationError() } },
            progress: { _ in reports += 1 }
        )) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(reports, 1)
    }
}
