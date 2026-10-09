import Foundation
import XCTest
@testable import MereRunCore

final class QwenImage21TurboTests: XCTestCase {
    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/QwenImage21Turbo")
    }

    private func decode<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: fixtures.appending(path: name)))
    }

    func testTurboHasSeparatePinnedInstallAndEightStepDefaults() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: QwenImage21Resources.turboModelID))
        XCTAssertEqual(spec.upstreamRepoId, "Qwen/Qwen-Image-2.1-Turbo")
        XCTAssertEqual(spec.hubFallback?.revision, QwenImage21Resources.turboRevision)
        XCTAssertEqual(spec.upstreamRevision, QwenImage21Resources.turboRevision)
        XCTAssertEqual(spec.validationKind, .qwenImage21)
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        let manifest = MereRunModelManifest.template(for: .qwenImage21Turbo)
        XCTAssertEqual(try ImageGenerationBackend(manifest: manifest), .qwenImage21)
        XCTAssertEqual(manifest.tier, .turbo)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.components?.tokenizer, .local(path: "processor"))
        XCTAssertEqual(Set(manifest.supports ?? []), [.txt2img, .referenceEdit])
        let options = ImageGenerationOptions(prompt: "A teapot", outputURL: URL(fileURLWithPath: "/tmp/teapot.png"))
        let sampling = ImageGenerationSampling.resolve(options, manifest: manifest)
        XCTAssertEqual(sampling.steps, 8)
        XCTAssertEqual(sampling.guidanceScale, 1)
        XCTAssertNil(sampling.sigmaShift)
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).isEmpty)
        let base = MereRunModelManifest.template(for: .qwenImage21)
        XCTAssertEqual(base.defaults?.steps, 40)
        XCTAssertEqual(base.tier, .latest)
        XCTAssertNotEqual(base.upstreamRepoId, manifest.upstreamRepoId)
    }

    func testSavedGridMatchesUpstreamSchedulerWithoutResolutionShift() throws {
        let pipeline = try decode("model_index.json", as: QwenImage21PipelineConfiguration.self)
        let scheduler = try decode("scheduler.json", as: QwenImage21Scheduler.self)
        // Diffusers PR #14950's independent regression asserts this exact grid plus terminal zero.
        let expected: [Float] = [1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0]
        for tokenCount in [256, 4096, 16384] {
            XCTAssertEqual(try scheduler.sigmas(steps: 8, tokenCount: tokenCount, sampleSigmas: pipeline.sampleSigmas), expected)
        }
        XCTAssertThrowsError(try scheduler.sigmas(steps: 40, tokenCount: 4096, sampleSigmas: pipeline.sampleSigmas))
        XCTAssertThrowsError(try scheduler.sigmas(steps: 8, tokenCount: 4096, shift: 3, sampleSigmas: pipeline.sampleSigmas))
        let invalidGrids: [[Float]] = [[], [1, 0], [1, 0.5, 0.5], [1, 0.7, 0.8], [1, .nan], [1, .infinity], [1.1, 0.5]]
        for invalid in invalidGrids {
            XCTAssertThrowsError(try scheduler.sigmas(steps: invalid.count, tokenCount: 4096, sampleSigmas: invalid))
        }
    }

    func testTurboPlanRejectsRecipeConflictsBeforeLoading() {
        let manifest = MereRunModelManifest.template(for: .qwenImage21Turbo)
        var options = ImageGenerationOptions(prompt: "A teapot", outputURL: URL(fileURLWithPath: "/tmp/teapot.png"), steps: 40)
        options.sigmaShift = 3
        let codes = Set(ImageGenerationPlan.issues(options, manifest: manifest).map(\.code))
        XCTAssertTrue(codes.isSuperset(of: ["steps_invalid", "sigma_shift_unsupported"]))
        options.steps = 8
        options.sigmaShift = nil
        options.referenceImages = Array(repeating: URL(fileURLWithPath: "/tmp/ref.png"), count: 11)
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).contains { $0.code == "reference_count_invalid" })
    }

    func testNativeLayerSchemasMatchTurboSafetensorsHeaders() throws {
        let shapes = try decode("weight-shapes.json", as: [String: [String: [Int]]].self)
        let transformer = try decode("transformer.json", as: QwenImage21TransformerConfig.self)
        let vae = try decode("vae.json", as: QwenImage21VAEConfig.self)
        XCTAssertEqual(QwenImage21Transformer.weightShapes(transformer), shapes["transformer"])
        XCTAssertEqual(QwenImage21VAE.weightShapes(vae), shapes["vae"])
        let encoder = try decode("text-encoder.json", as: Qwen3VLEmbeddingRootConfig.self)
        XCTAssertEqual(encoder.textConfig.hiddenSize, transformer.contextInDim)
        XCTAssertEqual(shapes["text_encoder"]?["model.language_model.embed_tokens.weight"],
                       [encoder.textConfig.vocabSize, encoder.textConfig.hiddenSize])
    }

    func testSavedScheduleAndCombinedProcessorAreRequiredWithoutWeights() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root.appending(path: "scheduler"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = QwenImage21Resources(rootURL: root)
        try FileManager.default.copyItem(at: fixtures.appending(path: "scheduler.json"), to: root.appending(path: "scheduler/scheduler_config.json"))
        try Data("{}".utf8).write(to: root.appending(path: "model_index.json"))
        XCTAssertThrowsError(try resources.samplingSchedule(steps: 8, tokenCount: 4096, shift: nil, requiresSavedSchedule: true))
        try Data(contentsOf: fixtures.appending(path: "model_index.json")).write(to: root.appending(path: "model_index.json"))
        XCTAssertEqual(try resources.samplingSchedule(steps: 8, tokenCount: 4096, shift: nil, requiresSavedSchedule: true).count, 9)
        XCTAssertThrowsError(try resources.samplingSchedule(steps: 4, tokenCount: 4096, shift: nil, requiresSavedSchedule: true))
        try FileManager.default.createDirectory(at: root.appending(path: "processor"), withIntermediateDirectories: true)
        let combined = root.appending(path: "processor/processor_config.json")
        try Data(contentsOf: fixtures.appending(path: "processor.json")).write(to: combined)
        XCTAssertFalse(resources.validate().contains(root.appending(path: "processor/preprocessor_config.json")))
        try FileManager.default.removeItem(at: combined)
        XCTAssertTrue(resources.validate().contains(root.appending(path: "processor/preprocessor_config.json")))
    }
}
