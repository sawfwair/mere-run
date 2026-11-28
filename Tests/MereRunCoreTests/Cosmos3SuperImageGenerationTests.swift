import Foundation
@testable import MereRunVideo
import MereRunContract
import XCTest
@testable import MereRunCore

/// `image generate` serves the distilled Cosmos3-Super checkpoint on the runtime
/// `video cosmos3 --mode text-to-image` uses.
final class Cosmos3SuperImageGenerationTests: XCTestCase {
    private let modelID = ModelResolver.ModelID.cosmos3SuperTextToImage
    private var manifest: MereRunModelManifest { .template(for: modelID) }

    func testManagedManifestSelectsTheCosmosBackendAndItsFourStepRecipe() throws {
        XCTAssertEqual(try ImageGenerationBackend(manifest: manifest), .cosmos3Super)
        let sampling = ImageGenerationSampling.resolve(options(), manifest: manifest)
        XCTAssertEqual(sampling.steps, 4)
        XCTAssertEqual(sampling.guidanceScale, 1)
        XCTAssertEqual(ModelFamilyIdentifier.imageGenerateFamily(manifest), "cosmos3-super")
        let family = try XCTUnwrap(
            MereRunCapabilityCatalog.imageGenerate.routing?.families.first { $0.id == "cosmos3-super" }
        )
        XCTAssertEqual(family.models, [modelID.rawValue])
        // The Edge checkpoint generates video and stays on `video cosmos3`.
        XCTAssertThrowsError(try ImageGenerationBackend(manifest: .template(for: .cosmos3EdgeMLX)))
    }

    func testPlanSendsThePublishedRecipeAndAcceptsANegativePrompt() throws {
        let root = try temporaryDirectory()
        var request = options()
        request.negativePrompt = "blurry"
        request.seed = 7
        let plan = try ImageGenerationPlan.resolve(request, modelRoot: root, manifest: manifest)
        XCTAssertEqual(plan.backend, .cosmos3Super)
        XCTAssertEqual(plan.request.steps, 4)
        XCTAssertEqual(plan.request.guidanceScale, 1)
        XCTAssertEqual(plan.request.seed, 7)
        XCTAssertEqual(plan.request.model, root.standardizedFileURL.path)
        XCTAssertTrue(plan.request.referenceImages.isEmpty)
    }

    func testPlanRejectsWhatTheDistilledCheckpointCannotDo() {
        let image = URL(fileURLWithPath: "/tmp/reference.png")
        let cases: [(String, (inout ImageGenerationOptions) -> Void)] = [
            ("input_mode_unsupported", { $0.inputImage = image }),
            ("references_unsupported", { $0.referenceImages = [image] }),
            ("steps_invalid", { $0.steps = 8 }),
            ("cfg_invalid", { $0.guidanceScale = 4 }),
            ("sigma_shift_unsupported", { $0.sigmaShift = 3 }),
            ("dimensions_invalid", { $0.width = 770 }),
            ("output_format_invalid", { $0.outputURL = URL(fileURLWithPath: "/tmp/image.jpg") }),
            ("lora_unsupported", { $0.loras = [ImageLoRAReference(raw: "style", reference: "style", scale: 1)] }),
        ]
        for (code, change) in cases {
            var request = options()
            change(&request)
            let codes = ImageGenerationPlan.issues(request, manifest: manifest).map(\.code)
            XCTAssertTrue(codes.contains(code), "Expected \(code), got \(codes)")
        }
        XCTAssertTrue(ImageGenerationPlan.issues(options(), manifest: manifest).isEmpty)
    }

    func testABoundCopyWithoutAManifestRunsOnTheManagedOne() throws {
        let root = try temporaryDirectory()
        let selection = ImageGenerationModelSelection(modelID.rawValue)
        XCTAssertEqual(try selection.loadManifest(at: root).engine, .cosmos3Super)

        // The other image backends require the manifest in the folder, so a missing one stays an error.
        for other in [ImageGenerationModelSelection("image-klein-nano"), ImageGenerationModelSelection(root.path)] {
            XCTAssertThrowsError(try other.loadManifest(at: root)) { error in
                XCTAssertTrue(error.localizedDescription.contains(MereRunModelManifest.filename))
            }
        }

        // A manifest in the folder always describes it.
        try MereRunModelManifest(id: "fixture", engine: .flux1, family: .flux1).write(to: root)
        XCTAssertEqual(try selection.loadManifest(at: root).id, "fixture")
    }

    func testOnlyCosmosRunsWithoutAnInstalledManifest() {
        for backend in ImageGenerationBackend.allCases {
            XCTAssertEqual(backend.runsWithoutInstalledManifest, backend == .cosmos3Super, "\(backend)")
        }
    }

    func testGeneratorRefusesAnIncompleteCheckpointBeforeLoadingWeights() async throws {
        let root = try temporaryDirectory()
        let request = GenerationRequest(
            prompt: "A red cube", width: 768, height: 768, steps: 4, guidanceScale: 1,
            outputURL: root.appendingPathComponent("image.png"), model: root.path
        )
        do {
            _ = try await Cosmos3SuperImageGenerator().generate(request, progressHandler: nil)
            XCTFail("Expected an incomplete checkpoint error")
        } catch let error as Cosmos3SuperImageGeneratorError {
            guard case .incompleteCheckpoint(let issues) = error else { return XCTFail("Unexpected error: \(error)") }
            XCTAssertFalse(issues.isEmpty)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputURL.path))
    }

    private func options() -> ImageGenerationOptions {
        ImageGenerationOptions(
            prompt: "A red cube on a clean neutral background",
            outputURL: URL(fileURLWithPath: "/tmp/image.png"), width: 768, height: 768
        )
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
