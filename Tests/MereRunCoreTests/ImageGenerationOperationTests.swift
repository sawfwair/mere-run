import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class ImageGenerationOperationTests: XCTestCase {
    func testSamplingUsesManifestDefaultsForEverySupportedBackend() throws {
        let options = ImageGenerationOptions(prompt: "A brass camera", outputURL: URL(fileURLWithPath: "/tmp/image.png"))
        let families: [(MereRunModelManifest.Family, MereRunModelManifest.Engine, ImageGenerationBackend)] = [
            (.flux1, .flux1, .flux1), (.klein, .flux2Klein, .flux2Klein), (.zimage, .zimageTurbo, .zImageTurbo),
            (.hidream, .hidreamO1, .hiDreamO1), (.senseNova, .senseNovaU15, .senseNovaU15),
            (.krea, .krea2, .krea2), (.ideogram, .ideogram4, .ideogram4), (.qwen, .qwenImageEdit, .qwenImageEdit)
        ]
        for (family, engine, backend) in families {
            let manifest = MereRunModelManifest(id: "fixture", engine: engine, family: family, defaults: .init(steps: 27, cfg: 3, sigmaShift: 2))
            let sampling = ImageGenerationSampling.resolve(options, manifest: manifest)
            XCTAssertEqual(try ImageGenerationBackend(manifest: manifest), backend)
            XCTAssertEqual(sampling.steps, family == .zimage ? 4 : 27)
            XCTAssertEqual(sampling.guidanceScale, family == .zimage ? 1 : 3)
            XCTAssertEqual(sampling.sigmaShift, 2)
        }
        let unknown = ImageGenerationSampling.resolve(options, manifest: nil)
        XCTAssertNil(unknown.steps)
        XCTAssertNil(unknown.guidanceScale)
    }

    func testModelSelectionPreservesLocalPathsAndKnownIDs() throws {
        let root = try temporaryDirectory()
        let local = ImageGenerationModelSelection(root.path)
        guard case .local = local else { return XCTFail("Expected local selection") }
        XCTAssertEqual(try local.resolveRoot().standardizedFileURL, root.standardizedFileURL)
        guard case .managed(.kleinNano) = ImageGenerationModelSelection("image-klein-nano") else {
            return XCTFail("Expected managed selection")
        }
        let unknown = ImageGenerationModelSelection(root.appendingPathComponent("missing-model").path)
        XCTAssertThrowsError(try unknown.resolveRoot()) { error in
            XCTAssertEqual((error as? ImageGenerationIssue)?.code, "model_unknown")
        }
    }

    func testTurboRecipeAndExplicitOverridesUseOneSamplingResolver() throws {
        var options = ImageGenerationOptions(
            prompt: "A camera", outputURL: URL(fileURLWithPath: "/tmp/image.png"),
            loras: try ImageLoRAReference.parse([ManagedAdapterCatalog.flux2DevTurboEightStepID], defaultScale: 1)
        )
        let manifest = MereRunModelManifest(id: "image-flux2-dev", family: .klein, defaults: .init(steps: 50, cfg: 4))
        let recipe = ImageGenerationSampling.resolve(options, manifest: manifest)
        XCTAssertEqual(recipe.sigmas, Flux2DevTurboRecipe.sigmas)
        XCTAssertEqual(recipe.steps, Flux2DevTurboRecipe.sigmas.count)
        XCTAssertEqual(recipe.guidanceScale, Flux2DevTurboRecipe.guidanceScale)
        options.sigmas = try ImageGenerationSampling.parseSigmas("1,0.5,0")
        options.steps = 2
        options.guidanceScale = 2.5
        let explicit = ImageGenerationSampling.resolve(options, manifest: manifest)
        XCTAssertEqual(explicit.sigmas, [1, 0.5])
        XCTAssertEqual(explicit.steps, 2)
        XCTAssertEqual(explicit.guidanceScale, 2.5)
        options.steps = 3
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).contains { $0.code == "sigma_step_count_mismatch" })
    }

    func testUnsupportedModesAndNonfiniteSettingsFailBeforePreparation() throws {
        let root = try temporaryDirectory()
        var options = ImageGenerationOptions(
            prompt: "A camera", outputURL: root.appendingPathComponent("new/result.png"),
            inputImage: root.appendingPathComponent("missing.png")
        )
        let manifest = MereRunModelManifest(id: "fixture", family: .flux1)
        XCTAssertThrowsError(try ImageGenerationPlan.resolve(options, modelRoot: root, manifest: manifest)) { error in
            XCTAssertEqual((error as? ImageGenerationIssue)?.code, "input_mode_unsupported")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: options.outputURL.deletingLastPathComponent().path))
        options.guidanceScale = .infinity
        options.width = Int.max
        options.height = Int.max
        options.strength = .nan
        let codes = Set(ImageGenerationPlan.issues(options).map(\.code))
        XCTAssertTrue(codes.isSuperset(of: ["cfg_invalid", "dimensions_invalid", "strength_invalid"]))
        XCTAssertThrowsError(try ImageGenerationSampling.parseSigmas("1,nan,0"))
        XCTAssertThrowsError(try ImageGenerationSampling.parseSigmas("0.2,0.8"))
    }

    func testPreviouslyIgnoredInputsAreRejectedByPreflightRules() {
        let output = URL(fileURLWithPath: "/tmp/image.png")
        let reference = URL(fileURLWithPath: "/tmp/reference.png")
        var options = ImageGenerationOptions(prompt: "A camera", outputURL: output, referenceImages: [reference])
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: .init(id: "zimage", family: .zimage))
            .contains { $0.code == "references_unsupported" })
        options.referenceImages = []
        options.loras = [.init(raw: "style", reference: "style.safetensors", scale: 1)]
        let families: [MereRunModelManifest.Family] = [.hidream, .senseNova, .qwen, .ideogram]
        for family in families {
            XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: .init(id: "fixture", family: family))
                .contains { $0.code == "lora_unsupported" })
        }
        options.loras = []
        options.width = 16
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: .init(id: "sensenova", family: .senseNova))
            .contains { $0.code == "dimensions_invalid" })
    }

    func testSuccessfulMaskedOperationPreservesPixelsAndReportsActualSeed() async throws {
        let plan = try maskedPlan()
        let recorder = ImageOperationRecorder()
        let id = UUID()
        let outcome = try await ImageGenerationOperation.execute(plan, id: id, eventHandler: recorder.record) { _, request, progress in
            let prepared = try XCTUnwrap(request.inputImage)
            XCTAssertTrue(FileManager.default.fileExists(atPath: prepared.path))
            recorder.prepared(prepared)
            progress?(GenerationProgress(stage: .denoising, stepIndex: 0, totalSteps: request.steps))
            try MediaImageIO.writePNG(
                try MediaImage(width: 2, height: 1, rgba8: [0, 0, 255, 255, 0, 0, 255, 255]), to: request.outputURL
            )
            return GenerationResult(outputURL: request.outputURL, seed: 42)
        }
        XCTAssertEqual(outcome.id, id)
        XCTAssertEqual(outcome.effectiveRequest.seed, 42)
        XCTAssertEqual(outcome.effectiveRequest.inputImage, plan.options.inputImage)
        XCTAssertEqual(outcome.result.outputURL, plan.request.outputURL)
        XCTAssertEqual(recorder.events, ["started", "progress", "succeeded"])
        let prepared = try XCTUnwrap(recorder.preparedURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: prepared.deletingLastPathComponent().path))
        let image = try MediaImageIO.decode(outcome.result.outputURL)
        XCTAssertEqual(image.rgba8, [255, 0, 0, 255, 0, 0, 255, 255])
    }

    func testExecutionWithoutObserversDoesNotEnableRuntimeProgress() async throws {
        let root = try temporaryDirectory()
        let plan = try ImageGenerationPlan.resolve(
            .init(prompt: "A camera", outputURL: root.appendingPathComponent("result.png")),
            modelRoot: root, manifest: .init(id: "fixture", family: .zimage)
        )
        _ = try await ImageGenerationOperation.execute(plan, executor: { _, request, progress in
            XCTAssertNil(progress)
            return GenerationResult(outputURL: request.outputURL, seed: 42)
        })
    }

    func testFailedGenerationReleasesPreparationAndEmitsOneTerminalEvent() async throws {
        let plan = try maskedPlan()
        let recorder = ImageOperationRecorder()
        do {
            _ = try await ImageGenerationOperation.execute(plan, eventHandler: recorder.record) { _, request, _ in
                recorder.prepared(try XCTUnwrap(request.inputImage))
                throw ImageGenerationIssue("fixture_failure", "Fixture runtime failed.")
            }
            XCTFail("Expected generation failure")
        } catch let error as ImageGenerationIssue {
            XCTAssertEqual(error.code, "fixture_failure")
        }
        XCTAssertEqual(recorder.events, ["started", "failed"])
        let prepared = try XCTUnwrap(recorder.preparedURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: prepared.deletingLastPathComponent().path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: plan.request.outputURL.path))
    }

    func testCancellationDuringGenerationReleasesPreparationAndStaysDistinctFromFailure() async throws {
        let plan = try maskedPlan()
        let recorder = ImageOperationRecorder()
        let task = Task.detached {
            try await ImageGenerationOperation.execute(plan, eventHandler: recorder.record) { _, request, _ in
                recorder.prepared(try XCTUnwrap(request.inputImage))
                try await Task.sleep(nanoseconds: 30_000_000_000)
                return GenerationResult(outputURL: request.outputURL, seed: 1)
            }
        }
        defer { task.cancel() }
        for _ in 0..<200 {
            if recorder.preparedURL != nil { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let prepared = try XCTUnwrap(recorder.preparedURL)
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError { }
        XCTAssertEqual(recorder.events, ["started", "cancelled"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: prepared.deletingLastPathComponent().path))
    }

    func testExecutionRechecksInputsChangedAfterPreflight() async throws {
        let plan = try maskedPlan()
        try FileManager.default.removeItem(at: XCTUnwrap(plan.options.inputImage))
        let recorder = ImageOperationRecorder()
        do {
            _ = try await ImageGenerationOperation.execute(plan, eventHandler: recorder.record) { _, request, _ in
                XCTFail("Unavailable input must fail before entering the runtime")
                return GenerationResult(outputURL: request.outputURL, seed: 1)
            }
            XCTFail("Expected missing input")
        } catch let error as ImageGenerationIssue {
            XCTAssertEqual(error.code, "input_missing")
        }
        XCTAssertEqual(recorder.events, ["started", "failed"])
    }

    func testDuplicateAdapterSymlinksAreRejected() throws {
        let root = try temporaryDirectory()
        let first = root.appendingPathComponent("first.safetensors")
        let alias = root.appendingPathComponent("alias.safetensors")
        try Data().write(to: first)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: first)
        let loras = try ImageLoRAReference.parse([first.path, alias.path], defaultScale: 1)
        XCTAssertThrowsError(try ImageGenerationPlan.resolveLoRAs(loras, baseModelID: "fixture")) { error in
            XCTAssertEqual((error as? ImageGenerationIssue)?.code, "lora_duplicate")
        }
    }

    private func maskedPlan() throws -> ImageGenerationPlan {
        let root = try temporaryDirectory()
        let input = root.appendingPathComponent("input.png")
        let mask = root.appendingPathComponent("mask.png")
        try MediaImageIO.writePNG(try MediaImage(width: 2, height: 1, rgba8: [255, 0, 0, 255, 0, 255, 0, 255]), to: input)
        try MediaImageIO.writePNG(try MediaImage(width: 2, height: 1, rgba8: [0, 0, 0, 255, 255, 255, 255, 255]), to: mask)
        let options = ImageGenerationOptions(
            prompt: "Blue", outputURL: root.appendingPathComponent("new/result.png"), width: 2, height: 1,
            inputImage: input, mask: mask, maskFeather: 0
        )
        let plan = try ImageGenerationPlan.resolve(options, modelRoot: root, manifest: .init(id: "fixture", family: .zimage))
        XCTAssertFalse(FileManager.default.fileExists(atPath: options.outputURL.deletingLastPathComponent().path))
        return plan
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}

private final class ImageOperationRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedEvents: [String] = []
    private var recordedURL: URL?

    var events: [String] { lock.withLock { recordedEvents } }
    var preparedURL: URL? { lock.withLock { recordedURL } }
    func prepared(_ url: URL) { lock.withLock { recordedURL = url } }

    func record(_ event: ImageGenerationEvent) {
        lock.withLock {
            switch event {
            case .started: recordedEvents.append("started")
            case .progress: recordedEvents.append("progress")
            case .succeeded: recordedEvents.append("succeeded")
            case .failed: recordedEvents.append("failed")
            case .cancelled: recordedEvents.append("cancelled")
            }
        }
    }
}
