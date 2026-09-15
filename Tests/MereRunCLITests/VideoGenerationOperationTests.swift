import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class VideoGenerationOperationTests: XCTestCase {
    func testAPIAndCLIKeepCompatibilityDefaults() throws {
        let output = URL(fileURLWithPath: "/tmp/video-defaults.mp4")
        let api = try APIVideoGeneration.options(
            APIServerContract.videoGenerationPlan(from: OpenAIVideoGenerationRequest(prompt: " harbor ")), outputURL: output
        )
        let cli = try VideoGenerate.parse([" harbor "]).makeGenerationOptions(outputURL: output)
        XCTAssertEqual(api.prompt, "harbor")
        XCTAssertEqual(cli.prompt, " harbor ")
        XCTAssertEqual(api.resolvedRequestedModel, ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue)
        XCTAssertEqual(cli.resolvedRequestedModel, ModelResolver.ModelID.ltxVideo23AVMLX.rawValue)
        let apiPlan = try VideoGenerationPlan(options: api, profile: .ltx25Distilled)
        let cliPlan = try VideoGenerationPlan(options: cli, profile: .ltx23Distilled)
        XCTAssertEqual(apiPlan.width, 768)
        XCTAssertEqual(apiPlan.height, 512)
        XCTAssertEqual(apiPlan.seed, 10)
        XCTAssertEqual(apiPlan.autoDuration?.minimumSeconds, 1)
        XCTAssertEqual(apiPlan.autoDuration?.maximumSeconds, 20)
        XCTAssertEqual(cliPlan.width, 768)
        XCTAssertEqual(cliPlan.height, 512)
        XCTAssertEqual(cliPlan.seed, 42)
        XCTAssertNil(cliPlan.autoDuration)
    }

    func testAPIJSONFieldsRemainLiteralAndUnknownOptionsAreClientErrors() throws {
        let output = URL(fileURLWithPath: "/tmp/video-literal-fields.mp4")
        let plan = try APIServerContract.videoGenerationPlan(from: OpenAIVideoGenerationRequest(
            prompt: "--help", model: "--literal-model", size: "512x320", num_frames: 9, seed: -42,
            quality: " FINAL ", output_mode: " VIDEO-ONLY "
        ))
        let settings = try APIVideoGeneration.options(plan, outputURL: output)
        XCTAssertEqual(settings.prompt, "--help")
        XCTAssertEqual(settings.model, "--literal-model")
        XCTAssertEqual(settings.seed, -42)
        XCTAssertEqual(settings.quality, .final)
        XCTAssertEqual(settings.outputMode, .videoOnly)
        let invalid = try APIServerContract.videoGenerationPlan(from: OpenAIVideoGenerationRequest(
            prompt: "harbor", options: ["--not-a-video-option"]
        ))
        XCTAssertThrowsError(try APIVideoGeneration.options(invalid, outputURL: output)) { error in
            guard case APIRequestValidationError.invalidField(let field, let message) = error else {
                return XCTFail("Expected an API argument error, got \(error)")
            }
            XCTAssertEqual(field, "options")
            XCTAssertTrue(message.contains("--not-a-video-option"))
        }
    }

    func testCLIAndAPIReachTheSamePreparedLTXRequest() async throws {
        let root = try makeSplitRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("output.mp4")
        let api = try apiPlan(root: root, numFrames: 26, seed: 17)
        let apiSettings = try APIVideoGeneration.options(api, outputURL: output)
        let cliSettings = try VideoGenerate.parse([
            " harbor ", "--model", root.path, "--width", "512", "--height", "320",
            "--num-frames", "26", "--fps", "24", "--seed", "17", "--output-mode", "video-only",
        ]).makeGenerationOptions(outputURL: output)
        for settings in [apiSettings, cliSettings] {
            let prepared = try await VideoGenerationOperation.prepare(settings, allowAutoDownload: false)
            guard case .ltx(let request) = prepared.input else { return XCTFail("Expected LTX input") }
            XCTAssertEqual(prepared.plan.profile, .ltx23Distilled)
            XCTAssertEqual(prepared.plan.ltxRoute, .splitDistilledVideo)
            XCTAssertEqual(prepared.modelRoot.path, root.path)
            let native = request.unifiedOptions()
            XCTAssertEqual(native.prompt, "harbor")
            XCTAssertEqual(native.width, 512)
            XCTAssertEqual(native.height, 320)
            XCTAssertEqual(native.numFrames, 25)
            XCTAssertEqual(native.fps, 24)
            XCTAssertEqual(native.seed, 17)
            XCTAssertNil(request.plan.autoDuration)
            XCTAssertEqual(native.distilledLoRAStrengthStage1, 0)
            XCTAssertEqual(native.distilledLoRAStrengthStage2, 1)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }

    func testAPIAdvancedOptionsUseTheSharedFullQualityRecipe() async throws {
        let root = try makeFull25Root()
        defer { try? FileManager.default.removeItem(at: root) }
        let api = try APIServerContract.videoGenerationPlan(from: OpenAIVideoGenerationRequest(
            prompt: "harbor", model: root.path, size: "1024x768", num_frames: 97, fps: 24,
            quality: "final", output_mode: "audio-video", options: ["--ltx-preset", "hq", "--a2v-steps", "20"]
        ))
        let settings = try APIVideoGeneration.options(api, outputURL: root.appendingPathComponent("out.mp4"))
        let prepared = try await VideoGenerationOperation.prepare(settings, allowAutoDownload: false)
        guard case .ltx(let request) = prepared.input else { return XCTFail("Expected LTX input") }
        let native = request.unifiedOptions()
        XCTAssertEqual(prepared.plan.profile, .ltx25Full)
        XCTAssertEqual(prepared.plan.ltxRoute, .unifiedAV)
        XCTAssertEqual(native.inferenceSteps, 15)
        XCTAssertEqual(native.sampler.mode, .res2s)
        XCTAssertEqual(native.distilledLoRAStrengthStage1, 0.25)
        XCTAssertEqual(native.distilledLoRAStrengthStage2, 0.5)
        XCTAssertEqual(native.numFrames, 97)
        XCTAssertEqual(native.width, 1_024)
        XCTAssertEqual(native.height, 768)
        XCTAssertEqual(native.seed, 10)
    }

    func testDeletedConditioningFailsBeforeRuntimeSetupOrExecution() async throws {
        let root = try makeSplitRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("source.png")
        try Data("image fixture".utf8).write(to: image)
        let output = root.appendingPathComponent("uncreated/output.mp4")
        let settings = try VideoGenerate.parse(["harbor", "--model", root.path, "--image", image.path])
            .makeGenerationOptions(outputURL: output)
        _ = try await VideoGenerationOperation.prepare(settings, allowAutoDownload: false)
        try FileManager.default.removeItem(at: image)
        do {
            _ = try await VideoGenerationOperation.execute(
                settings, allowAutoDownload: false,
                prepareRuntime: { XCTFail("A deleted input must block runtime setup") },
                executor: { _, _ in XCTFail("A deleted input must block execution"); throw FixtureError.unexpected }
            )
            XCTFail("Expected missing input")
        } catch VideoGenerationError.invalidInput(let message) {
            XCTAssertTrue(message.contains("Image file not found"))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path))
    }

    func testInvalidDurationFailsBeforeMissingModelResolution() async throws {
        let output = URL(fileURLWithPath: "/tmp/video-invalid-\(UUID().uuidString)/out.mp4")
        let settings = try VideoGenerate.parse(["harbor", "--model-root", "/missing-video-model", "--duration", "inf"])
            .makeGenerationOptions(outputURL: output)
        do {
            _ = try await VideoGenerationOperation.execute(
                settings, allowAutoDownload: false,
                prepareRuntime: { XCTFail("Invalid settings must block runtime setup") },
                executor: { _, _ in XCTFail("Invalid settings must block execution"); throw FixtureError.unexpected }
            )
            XCTFail("Expected duration rejection")
        } catch let issue as VideoGenerationIssue {
            XCTAssertEqual(issue.id, "duration_invalid")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path))
    }

    func testAPIExecutionUsesOneCallerAdmissionAndPreservesArtifactProof() async throws {
        let root = try makeSplitRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let plan = try apiPlan(root: root)
        let output = root.appendingPathComponent("out.mp4")
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let outcome = try await withVFXRequestAdmission(using: admission) {
            try await APIVideoGeneration.generate(plan, outputURL: output, prepareRuntime: {}, executor: { request, eventHandler in
                let snapshot = await admission.snapshot()
                XCTAssertEqual(snapshot.activeRequests, 1)
                XCTAssertEqual(snapshot.totalAdmittedRequests, 1)
                XCTAssertNil(eventHandler, "HTTP generation has no command presentation")
                XCTAssertEqual(request.plan.seed, 42)
                try Data("fixture MP4".utf8).write(to: request.plan.options.outputURL)
                return VideoGenerationOutcome(primaryURL: request.plan.options.outputURL)
            })
        }
        XCTAssertEqual(outcome.primaryURL, output)
        let response = try APIServerContract.videoGenerationResponse(outputURL: output, plan: plan)
        XCTAssertEqual(response.artifact.byte_count, 11)
        XCTAssertEqual(response.artifact.sha256, try ModelArtifactPin.fileSHA256(output))
        let snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 1)
        XCTAssertEqual(snapshot.totalCompletedRequests, 1)
    }

    func testAPIMissingArtifactFailsAndReleasesAdmission() async throws {
        let root = try makeSplitRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let plan = try apiPlan(root: root)
        let output = root.appendingPathComponent("missing.mp4")
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        do {
            _ = try await withVFXRequestAdmission(using: admission) {
                try await APIVideoGeneration.generate(plan, outputURL: output, prepareRuntime: {}, executor: { _, _ in
                    VideoGenerationOutcome(primaryURL: output)
                })
            }
            XCTFail("Expected missing artifact rejection")
        } catch APIRequestValidationError.invalidField(let field, _) {
            XCTAssertEqual(field, "output")
        }
        let snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalCompletedRequests, 1)
    }

    func testCancellationDoesNotReturnAnUncooperativeExecutorResult() async throws {
        let root = try makeSplitRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let plan = try apiPlan(root: root)
        let output = root.appendingPathComponent("cancelled.mp4")
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let task = Task {
            try await withVFXRequestAdmission(using: admission) {
                try await APIVideoGeneration.generate(plan, outputURL: output, prepareRuntime: {}, executor: { _, _ in
                    withUnsafeCurrentTask { $0?.cancel() }
                    return VideoGenerationOutcome(primaryURL: output)
                })
            }
        }
        do {
            _ = try await task.value
            XCTFail("Cancellation must win over a completed executor")
        } catch is CancellationError {
            // The operation checks cancellation before accepting the result.
        }
        let snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalCompletedRequests, 0)
        XCTAssertEqual(snapshot.totalCancelledRequests, 1)
    }

    func testQuietVideoProgressKeepsTerminalEventsAndSuppressesDiagnostics() throws {
        let capture = VideoEventCapture()
        let presentation = CLIVideoGenerationPresentation(quiet: true, progressJSON: true, write: { capture.append($0) })
        let handler = try XCTUnwrap(presentation.eventHandler)
        handler(.diagnostic("This diagnostic must stay suppressed.\n"))
        handler(.progress(stage: "denoising", step: 0, totalSteps: 2))
        handler(.progress(stage: "decoding", step: 0, totalSteps: 1))
        handler(.progressFinished)
        handler(.progressFinished)
        XCTAssertEqual(capture.lines, [
            #"{"event":"progress","stage":"denoising","step":0,"total_steps":2}"# + "\n",
            #"{"event":"progress","stage":"denoising","step":2,"total_steps":2}"# + "\n",
            #"{"event":"progress","stage":"decoding","step":0,"total_steps":1}"# + "\n",
            #"{"event":"progress","stage":"decoding","step":1,"total_steps":1}"# + "\n",
        ])
        XCTAssertNil(CLIVideoGenerationPresentation(quiet: true, progressJSON: false).eventHandler)
    }

    private enum FixtureError: Error { case unexpected }

    private func apiPlan(root: URL, numFrames: Int = 9, seed: Int = 42) throws -> APIServerContract.VideoGenerationPlan {
        try APIServerContract.videoGenerationPlan(from: OpenAIVideoGenerationRequest(
            prompt: " harbor ", model: root.path, size: "512x320", num_frames: numFrames,
            fps: 24, seed: seed, output_mode: "video-only"
        ))
    }

    private func makeSplitRoot() throws -> URL {
        let root = try makeRoot(files: ["split_model.json", "transformer-distilled.safetensors", "vae_decoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors"])
        try Data(#"{"model_version":"2.3"}"#.utf8).write(to: root.appendingPathComponent("config.json"))
        return root
    }

    private func makeFull25Root() throws -> URL {
        try makeRoot(files: LTX25Resources.fullRequiredRelativePaths)
    }

    private func makeRoot(files: [String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("video-operation-\(UUID().uuidString)")
        for path in files {
            let file = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("fixture".utf8).write(to: file)
        }
        return root
    }
}

private final class VideoEventCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String] = []

    func append(_ value: String) {
        lock.lock()
        defer { lock.unlock() }
        values.append(value)
    }

    var lines: [String] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}
