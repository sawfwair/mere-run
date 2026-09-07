import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ImageGenerationAdapterTests: XCTestCase {
    func testCLIAndAPIResolveEquivalentExplicitSettingsAcrossImageBackends() throws {
        let root = try temporaryDirectory()
        let input = root.appendingPathComponent("input.png")
        try Data("fixture".utf8).write(to: input)
        let output = root.appendingPathComponent("result.png")
        let families: [(MereRunModelManifest.Family, MereRunModelManifest.Engine)] = [
            (.flux1, .flux1), (.klein, .flux2Klein), (.zimage, .zimageTurbo), (.hidream, .hidreamO1),
            (.senseNova, .senseNovaU15), (.krea, .krea2), (.ideogram, .ideogram4), (.qwen, .qwenImageEdit)
        ]
        for (family, engine) in families {
            let manifest = MereRunModelManifest(
                id: "fixture", engine: engine, family: family,
                defaults: .init(steps: 27, cfg: 3, sigmaShift: 2)
            )
            try manifest.write(to: root)
            let inputImage = family == .qwen ? input : nil
            var argv = [
                "--prompt", "A brass camera", "--model", root.path, "--output", output.path,
                "--width", "512", "--height", "512", "--steps", "12", "--cfg", "2.5", "--seed", "42"
            ]
            if let inputImage { argv += ["--input", inputImage.path] }
            let command = try ImageGenerate.parse(argv)
            let cli = try MereRunCore.ImageGenerationPlan.resolve(
                command.operationOptions(outputURL: output), modelRoot: root, manifest: manifest
            )
            let api = try apiPlan(steps: 12, guidance: 2.5, input: inputImage)
                .operationPlan(modelRoot: root, outputURL: output, manifest: manifest)
            XCTAssertEqual(cli.request, api.request, "Equivalent explicit settings for \(family)")
            XCTAssertEqual(cli.backend, api.backend)
            let preflight = command.makePreflightEnvelope(outputURL: output)
            XCTAssertEqual(preflight.status, .ok, "Preflight for \(family)")
            XCTAssertEqual(preflight.result.plan.effectiveSteps, cli.request.steps)
            XCTAssertEqual(preflight.result.plan.effectiveCFGScale, cli.request.guidanceScale)
            XCTAssertEqual(preflight.result.plan.effectiveSigmaShift, cli.request.sigmaShift.map(Double.init))
        }
    }

    func testAPIV1KleinCompatibilityIsExplicit() throws {
        let root = try temporaryDirectory()
        let input = root.appendingPathComponent("input.png")
        try Data("fixture".utf8).write(to: input)
        let output = root.appendingPathComponent("result.png")
        let manifest = MereRunModelManifest(id: "fixture", family: .klein, defaults: .init(steps: 28, cfg: 4))
        let api = try apiPlan(input: input).operationPlan(modelRoot: root, outputURL: output, manifest: manifest)
        let command = try ImageGenerate.parse(["--prompt", "A brass camera", "--input", input.path])
        let cli = try MereRunCore.ImageGenerationPlan.resolve(command.operationOptions(outputURL: output), modelRoot: root, manifest: manifest)
        XCTAssertEqual(api.request.steps, 4)
        XCTAssertEqual(api.request.guidanceScale, 1)
        XCTAssertEqual(api.request.inputImage, input)
        XCTAssertEqual(api.request.referenceStrength, 0)
        XCTAssertEqual(cli.request.steps, 28)
        XCTAssertEqual(cli.request.guidanceScale, 4)
        XCTAssertNil(cli.request.inputImage)
        XCTAssertEqual(cli.request.referenceImages, [input])
        XCTAssertEqual(cli.request.referenceStrength, 0.75)
    }

    func testQwenAPILegacyDefaultsAndWholeImageMaskCompatibility() throws {
        let root = try temporaryDirectory()
        let input = root.appendingPathComponent("input.png")
        try Data("fixture".utf8).write(to: input)
        let output = root.appendingPathComponent("result.png")
        let manifest = MereRunModelManifest(id: "fixture", engine: .qwenImageEdit, family: .qwen)
        let api = APIServerContract.ImageGenerationPlan(
            modelID: "fixture", prompt: "A brass camera", width: 512, height: 512, responseFormat: "url",
            seed: 42, negativePrompt: nil, steps: nil, guidanceScale: nil, inputImage: input,
            maskImage: root.appendingPathComponent("compatibility-mask.png"), strength: nil
        )
        let operation = try api.operationPlan(modelRoot: root, outputURL: output, manifest: manifest, qwenEditDefaults: true)
        XCTAssertEqual(operation.request.steps, 20)
        XCTAssertEqual(operation.request.guidanceScale, 4)
        XCTAssertNil(operation.options.mask)
        XCTAssertEqual(operation.request.inputImage, input)
        let legacy = try api.operationPlan(
            modelRoot: root, outputURL: output, manifest: .init(id: "fixture"), qwenEditDefaults: true
        )
        XCTAssertEqual(legacy.backend, .qwenImageEdit)
        XCTAssertEqual(legacy.request.steps, 20)
    }

    func testCLIAndPreflightRejectUnsupportedSigmasBeforeLoadingWeights() async throws {
        let root = try temporaryDirectory()
        let manifest = MereRunModelManifest(id: "fixture", family: .zimage)
        try manifest.write(to: root)
        let output = root.appendingPathComponent("new/result.png")
        let command = try ImageGenerate.parse([
            "--prompt", "A brass camera", "--model", root.path,
            "--sigmas", "1,0.5", "--output", output.path
        ])
        let preflight = command.makePreflightEnvelope(outputURL: output)
        XCTAssertEqual(preflight.status, .blocked)
        XCTAssertTrue(preflight.diagnostics.contains { $0.id == "sigma_model_unsupported" })
        do {
            try await command.run()
            XCTFail("Unsupported sigmas must fail before loading weights")
        } catch let error as ImageGenerationIssue {
            XCTAssertEqual(error.code, "sigma_model_unsupported")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path))
    }

    func testPreflightUsesExecutionConditioningModeForSenseNova() throws {
        let root = try temporaryDirectory()
        let manifest = MereRunModelManifest(id: "fixture", family: .senseNova)
        try manifest.write(to: root)
        let input = root.appendingPathComponent("input.png")
        try Data("fixture".utf8).write(to: input)
        let command = try ImageGenerate.parse(["--prompt", "A brass camera", "--model", root.path, "--input", input.path])
        let output = root.appendingPathComponent("result.png")
        let preflight = command.makePreflightEnvelope(outputURL: output)
        XCTAssertEqual(preflight.result.plan.inputMode, "image_to_image")
        XCTAssertEqual(preflight.status, .ok)
    }

    func testMissingInputKeepsOneActionablePreflightDiagnostic() throws {
        let root = try temporaryDirectory()
        try MereRunModelManifest(id: "fixture", family: .zimage).write(to: root)
        let command = try ImageGenerate.parse([
            "--prompt", "A camera", "--model", root.path, "--input", root.appendingPathComponent("missing.png").path
        ])
        let preflight = command.makePreflightEnvelope(outputURL: root.appendingPathComponent("result.png"))
        XCTAssertEqual(preflight.status, .blocked)
        XCTAssertEqual(preflight.diagnostics.filter { $0.severity == .blocker }.map(\.id), ["input_image_missing"])
    }

    private func apiPlan(steps: Int? = nil, guidance: Double? = nil, input: URL? = nil) -> APIServerContract.ImageGenerationPlan {
        APIServerContract.ImageGenerationPlan(
            modelID: "fixture", prompt: "A brass camera", width: 512, height: 512, responseFormat: "url",
            seed: 42, negativePrompt: nil, steps: steps, guidanceScale: guidance, inputImage: input, strength: nil
        )
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
