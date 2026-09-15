import Foundation
import XCTest
@testable import MereRunCore

final class ImageLoRATrainingOperationTests: XCTestCase {
    func testKreaPlanPreservesRecipeAndSyntheticConfiguration() throws {
        let fixture = try makeFixture(family: .krea)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        var options = fixture.options
        options.recipe = "krea-cinematic-style"
        options.syntheticSamples = 2
        options.baseQuantizationBits = 4
        options.lite = true
        options.seed = 7
        let plan = try ImageLoRATrainingPlan.resolve(options)
        guard case .krea(let examples, let config) = plan.training else { return XCTFail("Expected Krea") }
        XCTAssertTrue(examples.isEmpty)
        XCTAssertNil(config.datasetRoot)
        XCTAssertEqual(config.width, 768)
        XCTAssertEqual(config.height, 416)
        XCTAssertEqual(config.trainingSteps, 200)
        XCTAssertEqual(config.learningRate, 0.0001)
        XCTAssertEqual(config.loraRank, 32)
        XCTAssertEqual(config.loraAlpha, 32)
        XCTAssertEqual(config.lrWarmupSteps, 20)
        XCTAssertTrue(config.useCosineScheduler)
        XCTAssertEqual(config.lrMinFactor, 0)
        XCTAssertFalse(config.useCompile)
        XCTAssertEqual(config.syntheticSampleCount, 2)
        XCTAssertEqual(config.baseQuantizationBits, 4)
        XCTAssertEqual(config.loraTargetSuffixes, Krea2LoRAInjector.liteTargetSuffixes)
        XCTAssertEqual(config.seed, 7)
        XCTAssertFalse(FileManager.default.fileExists(atPath: plan.outputURL.deletingLastPathComponent().path))
    }

    func testKleinPlanPreservesRankPresetResumeAndBenchmarkInputs() throws {
        let fixture = try makeFixture(family: .klein)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        var options = fixture.options
        let resume = fixture.root.appendingPathComponent("resume.safetensors")
        try Data("optimizer state fixture".utf8).write(to: resume)
        options.resumeFrom = resume.path
        options.loraRankPreset = "flux2-style-128"
        options.gradientCheckpointing = true
        options.progressive = true
        options.lowRam = true
        options.benchmarkSteps = 3
        options.benchmarkWarmupSteps = 2
        options.timestepSampling = "logitNormal"
        options.timestepLossWeighting = "weighted"
        options.lossWeighting = "minSNR"
        options.timestepLow = 10
        options.timestepHigh = 900
        options.lrWarmupSteps = 4
        options.lrMinFactor = 0.2
        options.adamWeightDecay = 0.03
        let plan = try ImageLoRATrainingPlan.resolve(options)
        guard case .klein(let examples, let config, let checkpoint, let sample) = plan.training else {
            return XCTFail("Expected Klein")
        }
        XCTAssertEqual(examples.map(\.caption), ["training caption"])
        XCTAssertEqual(config.datasetRoot, options.data)
        XCTAssertEqual(config.loraRank, 128)
        XCTAssertEqual(config.loraAlpha, 64)
        XCTAssertEqual(config.loraTargetRankSuffixes?[".attn.to_q"], 128)
        XCTAssertEqual(checkpoint, resume)
        XCTAssertNil(sample)
        XCTAssertTrue(plan.isBenchmark)
        XCTAssertEqual(config.trainingSteps, 5)
        XCTAssertEqual(config.benchmarkSteps, 3)
        XCTAssertEqual(config.benchmarkWarmupSteps, 2)
        XCTAssertTrue(config.gradientCheckpointing)
        XCTAssertFalse(config.useCompile)
        XCTAssertTrue(config.progressive)
        XCTAssertTrue(config.lowRam)
        XCTAssertEqual(config.timestepSampling, .logitNormal)
        XCTAssertEqual(config.timestepLossWeighting, .weighted)
        XCTAssertEqual(config.lossWeighting, .minSNR)
        XCTAssertEqual(config.timestepLow, 10)
        XCTAssertEqual(config.timestepHigh, 900)
        XCTAssertEqual(config.lrWarmupSteps, 4)
        XCTAssertEqual(config.lrMinFactor, 0.2)
        XCTAssertEqual(config.adamWeightDecay, 0.03)
    }

    func testKleinRecipeAndPreviewPlanUseCaptionFallbackAndExplicitOverrides() throws {
        let fixture = try makeFixture(family: .klein)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        var options = fixture.options
        options.recipe = "klein-fast-style"
        options.widthOverride = 512
        options.heightOverride = 256
        options.rankOverride = 8
        options.sampleInterval = 3
        options.sampleModel = options.model
        options.sampleSteps = 4
        options.sampleGuidanceScale = 1.5
        options.sampleLoRAScale = 0.7
        options.sampleSeed = 11
        let plan = try ImageLoRATrainingPlan.resolve(options)
        guard case .klein(_, let config, _, let sample) = plan.training else { return XCTFail("Expected Klein") }
        XCTAssertEqual(config.loraRank, 8)
        XCTAssertEqual(config.loraAlpha, 8)
        XCTAssertEqual(config.loraTargetRanks?.count, 120)
        XCTAssertEqual(config.loraTargetRanks?["x_embedder"], 8)
        XCTAssertEqual(config.checkpointInterval, 250)
        XCTAssertEqual(config.maxResolution, 512)
        XCTAssertTrue(config.lowRam)
        XCTAssertFalse(config.useCompile)
        let preview = try XCTUnwrap(sample)
        XCTAssertEqual(preview.prompt, "training caption")
        XCTAssertEqual(preview.modelPath, options.model)
        XCTAssertEqual(preview.width, 512)
        XCTAssertEqual(preview.height, 256)
        XCTAssertEqual(preview.steps, 4)
        XCTAssertEqual(preview.guidanceScale, 1.5)
        XCTAssertEqual(preview.loraScale, 0.7)
        XCTAssertEqual(preview.seed, 11)
    }

    func testPlanRetainsCaptionSnapshotAndExcludesPreviews() throws {
        let fixture = try makeFixture(family: .krea)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let data = URL(fileURLWithPath: try XCTUnwrap(fixture.options.data))
        try Data().write(to: data.appendingPathComponent("preview1.png"))
        try "preview".write(to: data.appendingPathComponent("preview1.txt"), atomically: true, encoding: .utf8)
        var options = fixture.options
        options.excludePreviewImages = true
        let plan = try ImageLoRATrainingPlan.resolve(options)
        try "changed".write(to: data.appendingPathComponent("one.txt"), atomically: true, encoding: .utf8)
        guard case .krea(let examples, let config) = plan.training else { return XCTFail("Expected Krea") }
        XCTAssertEqual(examples.map(\.caption), ["training caption"])
        XCTAssertEqual(config.datasetRoot, data.path)
    }

    func testFamilyAndTargetValidationRunWithoutLoadingWeights() throws {
        let krea = try makeFixture(family: .krea)
        defer { try? FileManager.default.removeItem(at: krea.root) }
        var invalid = krea.options
        invalid.checkpointInterval = 2
        XCTAssertThrowsError(try ImageLoRATrainingPlan.resolve(invalid)) { error in
            XCTAssertEqual(error.localizedDescription, "--checkpoint-interval is only supported for FLUX.2 Klein LoRA training")
        }
        invalid = krea.options
        invalid.samplePrompt = "preview"
        XCTAssertThrowsError(try ImageLoRATrainingPlan.resolve(invalid)) { error in
            XCTAssertEqual(error.localizedDescription, "Klein training options require a FLUX.2 Klein base model.")
        }
        let klein = try makeFixture(family: .klein)
        defer { try? FileManager.default.removeItem(at: klein.root) }
        invalid = klein.options
        invalid.baseQuantizationBits = 4
        XCTAssertThrowsError(try ImageLoRATrainingPlan.resolve(invalid)) { error in
            XCTAssertEqual(error.localizedDescription, "--base-quantization-bits is only supported for Krea 2 LoRA training")
        }
        invalid = klein.options
        invalid.timestepSampling = "unsupported"
        XCTAssertThrowsError(try ImageLoRATrainingPlan.resolve(invalid)) { error in
            XCTAssertEqual(error.localizedDescription, "Unsupported --timestep-sampling 'unsupported'")
        }
    }

    func testOperationForwardsPreparedInputsAndReturnsSavedOrBenchmarkOutcome() async throws {
        for benchmark in [false, true] {
            let fixture = try makeFixture(family: .klein)
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            var options = fixture.options
            if benchmark { options.benchmarkSteps = 2 }
            let plan = try ImageLoRATrainingPlan.resolve(options)
            let progressReceived = expectation(description: "Training progress forwarded")
            let outcome = try await ImageLoRATrainingOperation.execute(plan, progress: { update in
                if case .klein = update { progressReceived.fulfill() }
            }, trainer: { received, progress in
                XCTAssertEqual(received.outputURL, plan.outputURL)
                XCTAssertEqual(received.modelRoot, plan.modelRoot)
                XCTAssertEqual(received.isBenchmark, benchmark)
                progress?(.klein(.init(stage: .saving, fraction: 1)))
                if !benchmark { try Data("adapter".utf8).write(to: received.outputURL) }
            })
            XCTAssertEqual(outcome, benchmark ? .benchmark : .saved(plan.outputURL))
            XCTAssertEqual(FileManager.default.fileExists(atPath: plan.outputURL.path), !benchmark)
            await fulfillment(of: [progressReceived], timeout: 1)
        }
    }

    func testCancellationCannotStartTrainerOrReturnSavedOutcome() async throws {
        for cancelBefore in [false, true] {
            let fixture = try makeFixture(family: .krea)
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let plan = try ImageLoRATrainingPlan.resolve(fixture.options)
            let trainer: ImageLoRATrainingOperation.Trainer = { _, _ in
                XCTAssertFalse(cancelBefore)
                withUnsafeCurrentTask { $0?.cancel() }
            }
            let task = Task {
                if cancelBefore { withUnsafeCurrentTask { $0?.cancel() } }
                return try await ImageLoRATrainingOperation.execute(plan, trainer: trainer)
            }
            do {
                _ = try await task.value
                XCTFail("Expected cancellation")
            } catch { XCTAssertTrue(error is CancellationError) }
            XCTAssertFalse(FileManager.default.fileExists(atPath: plan.outputURL.path))
            if cancelBefore {
                XCTAssertFalse(FileManager.default.fileExists(atPath: plan.outputURL.deletingLastPathComponent().path))
            }
        }
    }

    private func makeFixture(family: MereRunModelManifest.Family) throws -> (root: URL, options: ImageLoRATrainingOptions) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let model = root.appendingPathComponent("model")
        let data = root.appendingPathComponent("data")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true)
        let manifest = MereRunModelManifest(
            id: family == .krea ? ModelResolver.ModelID.krea2Raw.rawValue : ModelResolver.ModelID.kleinBase9B.rawValue,
            family: family, variant: .base, precision: .bf16, supports: [.loraTraining]
        )
        try JSONEncoder().encode(manifest).write(to: model.appendingPathComponent(MereRunModelManifest.filename))
        try Data().write(to: data.appendingPathComponent("one.png"))
        try "training caption".write(to: data.appendingPathComponent("one.txt"), atomically: true, encoding: .utf8)
        var options = ImageLoRATrainingOptions(data: data.path, output: root.appendingPathComponent("output/adapter.safetensors").path)
        options.model = model.path
        return (root, options)
    }
}
