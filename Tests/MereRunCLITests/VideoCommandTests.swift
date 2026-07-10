import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VideoCommandTests: XCTestCase {
    private var temporaryDirectories: [URL] = []

    override func tearDown() {
        for directory in temporaryDirectories {
            try? FileManager.default.removeItem(at: directory)
        }
        temporaryDirectories.removeAll()
        super.tearDown()
    }

    func testVideoCommandExposesGenerateAndExportLatents() {
        let commandNames = Set(Video.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(commandNames, Set(["generate", "export-latents"]))
    }

    func testVideoGenerateParsesDefaults() throws {
        let cmd = try VideoGenerate.parse([
            "a cinematic drone flythrough"
        ])

        XCTAssertEqual(cmd.prompt, "a cinematic drone flythrough")
        XCTAssertEqual(cmd.variant, .distilled)
        XCTAssertNil(cmd.width)
        XCTAssertNil(cmd.height)
        XCTAssertNil(cmd.numFrames)
        XCTAssertNil(cmd.duration)
        XCTAssertEqual(cmd.fps, 24)
        XCTAssertEqual(cmd.imageStrength, 1.0)
        XCTAssertNil(cmd.endImage)
        XCTAssertEqual(cmd.endImageStrength, 1.0)
        XCTAssertNil(cmd.modelRoot)
        XCTAssertEqual(cmd.steps, 40)
        XCTAssertEqual(cmd.guidanceScale, 3)
        XCTAssertEqual(cmd.shift, 3)
        XCTAssertFalse(cmd.batchCFG)
        XCTAssertFalse(cmd.temporalProbe)
        XCTAssertEqual(cmd.temporalProbeStep, 4)
        XCTAssertFalse(cmd.refiner)
        XCTAssertEqual(cmd.refinerSteps, 8)
        XCTAssertEqual(cmd.refinerGuidanceScale, 3)
        XCTAssertEqual(cmd.refinerShift, 3)
        XCTAssertEqual(cmd.refinerThreshold, 0.85)
        XCTAssertEqual(cmd.refinerSigmaTailSteps, 2)
        XCTAssertFalse(cmd.refinerBatchCFG)
        XCTAssertFalse(cmd.preflight)
        XCTAssertFalse(cmd.json)
    }

    func testVideoGenerateParsesPreflightJSONFlags() throws {
        let cmd = try VideoGenerate.parse([
            "a cinematic drone flythrough",
            "--preflight",
            "--json",
        ])

        XCTAssertTrue(cmd.preflight)
        XCTAssertTrue(cmd.json)
    }

    func testVideoGenerateParsesStartAndEndKeyframes() throws {
        let cmd = try VideoGenerate.parse([
            "a flower opens from bud to bloom",
            "--image", "/tmp/start.png",
            "--image-strength", "0.85",
            "--end-image", "/tmp/end.png",
            "--end-image-strength", "0.7",
        ])

        XCTAssertEqual(cmd.image, "/tmp/start.png")
        XCTAssertEqual(cmd.imageStrength, 0.85, accuracy: 0.0001)
        XCTAssertEqual(cmd.endImage, "/tmp/end.png")
        XCTAssertEqual(cmd.endImageStrength, 0.7, accuracy: 0.0001)
    }

    func testVideoGenerateRejectsEndImageWithoutStartImage() async throws {
        let cmd = try VideoGenerate.parse([
            "a car drives from dawn into sunset",
            "--end-image", "/tmp/end.png",
        ])

        do {
            try await cmd.run()
            XCTFail("Expected --end-image without --image to fail validation.")
        } catch {
            let message = "\(error) \(error.localizedDescription)"
            XCTAssertTrue(message.contains("--end-image requires --image"))
        }
    }

    func testVideoGenerateParsesDurationOverride() throws {
        let cmd = try VideoGenerate.parse([
            "dialogue with background music",
            "--variant", "unified-av",
            "--duration", "15",
            "--fps", "24",
        ])

        XCTAssertEqual(cmd.variant, .unifiedAV)
        XCTAssertEqual(cmd.duration, 15)
        XCTAssertEqual(cmd.fps, 24)
    }

    func testNearestLTXFrameCountUsesClosestLegalFrameCount() {
        XCTAssertEqual(nearestLTXFrameCount(duration: 15, fps: 24), 361)
        XCTAssertEqual(nearestLTXFrameCount(duration: 5, fps: 24), 121)
        XCTAssertEqual(nearestLTXFrameCount(duration: 15, fps: 8), 121)
    }

    func testLingBotModelSelectsNativeLingBotVariant() throws {
        let cmd = try VideoGenerate.parse([
            "a robot folds a towel",
            "--model", "video-lingbot-dense-1.3b",
            "--steps", "12",
            "--guidance-scale", "2.5",
            "--shift", "3.5",
        ])

        XCTAssertEqual(cmd.variant, .distilled)
        XCTAssertEqual(cmd.effectiveVariant, .lingbot)
        XCTAssertEqual(cmd.steps, 12)
        XCTAssertEqual(cmd.guidanceScale, 2.5)
        XCTAssertEqual(cmd.shift, 3.5)

        let moe = try VideoGenerate.parse([
            "a robot folds a towel",
            "--model", LingBotVideoMoEQuantizer.defaultOutputModelID,
        ])
        XCTAssertEqual(moe.effectiveVariant, .lingbot)
    }

    func testLingBotRefinerParsesReleasedDefaultsAndOverrides() throws {
        let cmd = try VideoGenerate.parse([
            "a robot folds a towel",
            "--model", LingBotVideoMoEQuantizer.defaultOutputModelID,
            "--refiner",
            "--refiner-width", "960",
            "--refiner-height", "544",
            "--refiner-steps", "10",
            "--refiner-guidance-scale", "2.5",
            "--refiner-shift", "4",
            "--refiner-threshold", "0.8",
            "--refiner-sigma-tail-steps", "3",
            "--batch-cfg",
            "--refiner-batch-cfg",
        ])

        XCTAssertTrue(cmd.refiner)
        XCTAssertEqual(cmd.refinerWidth, 960)
        XCTAssertEqual(cmd.refinerHeight, 544)
        XCTAssertEqual(cmd.refinerSteps, 10)
        XCTAssertEqual(cmd.refinerGuidanceScale, 2.5)
        XCTAssertEqual(cmd.refinerShift, 4)
        XCTAssertEqual(cmd.refinerThreshold, 0.8)
        XCTAssertEqual(cmd.refinerSigmaTailSteps, 3)
        XCTAssertTrue(cmd.batchCFG)
        XCTAssertTrue(cmd.refinerBatchCFG)
    }

    func testLingBotTemporalProbeParsesAndDisablesRefinerInPreflightPlan() throws {
        let modelRoot = try makeValidLingBotModelRoot()
        let output = makeTempOutput(name: "probe.mp4")
        let cmd = try VideoGenerate.parse([
            "a robot folds a towel",
            "--model-root", modelRoot.path,
            "--variant", "lingbot",
            "--temporal-probe",
            "--temporal-probe-step", "6",
            "--refiner",
            "--output", output.path,
            "--preflight",
            "--json",
        ])

        XCTAssertTrue(cmd.temporalProbe)
        XCTAssertEqual(cmd.temporalProbeStep, 6)
        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )
        XCTAssertTrue(envelope.result.plan.temporalProbe)
        XCTAssertEqual(envelope.result.plan.temporalProbeStep, 6)
        XCTAssertFalse(envelope.result.plan.refiner)
        XCTAssertTrue(
            envelope.actions.first { $0.id == "start-video-generation" }?
                .command?.argv.contains("--temporal-probe") == true
        )
    }

    func testLingBotLegalOneShotGeometryIsNotBlockedByNonUpstreamHeuristics() throws {
        let modelRoot = try makeValidLingBotModelRoot()
        let output = makeTempOutput(name: "unsafe-one-shot.mp4")
        let cmd = try VideoGenerate.parse([
            "a robot folds a towel",
            "--model-root", modelRoot.path,
            "--variant", "lingbot",
            "--width", "320",
            "--height", "192",
            "--duration", "6",
            "--fps", "24",
            "--output", output.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )
        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.plan.resolvedNumFrames, 145)
        XCTAssertFalse(envelope.diagnostics.contains { $0.id == "lingbot_geometry_unstable" })
    }

    func testLingBotReferencePreflightReportsQuadraticAttentionWork() throws {
        let modelRoot = try makeValidLingBotModelRoot()
        let output = makeTempOutput(name: "reference-work.mp4")
        let cmd = try VideoGenerate.parse([
            "a robot folds a towel",
            "--model-root", modelRoot.path,
            "--variant", "lingbot",
            "--width", "832",
            "--height", "480",
            "--num-frames", "121",
            "--batch-cfg",
            "--output", output.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )
        XCTAssertEqual(envelope.result.plan.videoTokenCount, 48_360)
        XCTAssertEqual(envelope.result.plan.cfgPassesPerStep, 2)
        XCTAssertTrue(envelope.result.plan.batchCFG)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "lingbot_global_attention_large" })
        XCTAssertTrue(
            envelope.actions.first { $0.id == "start-video-generation" }?
                .command?.argv.contains("--batch-cfg") == true
        )
    }

    func testNearestLingBotFrameCountUsesClosestFourNPlusOneCount() {
        XCTAssertEqual(nearestLingBotFrameCount(duration: 5, fps: 24), 121)
        XCTAssertEqual(nearestLingBotFrameCount(duration: 1, fps: 24), 25)
        XCTAssertEqual(nearestLingBotFrameCount(duration: 0.2, fps: 24), 5)
        XCTAssertEqual(nearestLingBotFrameCount(duration: 5.1, fps: 24), 125)
    }

    func testLingBotPromptJSONSuppliesCaptionAndDurationWithoutPositionalPrompt() throws {
        let modelRoot = try makeValidLingBotModelRoot()
        let output = makeTempOutput(name: "prompt-json.mp4")
        let promptURL = try makeTempDirectory().appendingPathComponent("prompt.json")
        try Data(#"{"caption":{"scene":"a robot folds a towel"},"duration":5}"#.utf8)
            .write(to: promptURL)
        let negativeURL = promptURL.deletingLastPathComponent().appendingPathComponent("negative.json")
        try Data("{ \"z\" : [\"blur\"], \"a\" : [] }".utf8).write(to: negativeURL)
        let cmd = try VideoGenerate.parse([
            "--prompt-json", promptURL.path,
            "--negative-prompt-json", negativeURL.path,
            "--model-root", modelRoot.path,
            "--variant", "lingbot",
            "--output", output.path,
            "--preflight",
            "--json",
        ])

        XCTAssertNil(cmd.prompt)
        XCTAssertEqual(cmd.promptJSON, promptURL.path)
        XCTAssertEqual(cmd.negativePromptJSON, negativeURL.path)
        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )
        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.request.promptJSON, promptURL.path)
        XCTAssertEqual(envelope.request.negativePromptJSON, negativeURL.path)
        XCTAssertEqual(envelope.request.negativePrompt, #"{"z":["blur"],"a":[]}"#)
        XCTAssertEqual(envelope.result.plan.requestedDurationSeconds, 5)
        XCTAssertEqual(envelope.result.plan.resolvedNumFrames, 121)
        XCTAssertEqual(envelope.result.plan.resolvedWidth, 320)
        XCTAssertEqual(envelope.result.plan.resolvedHeight, 192)
    }

    func testLingBotRefinerPlanUsesReleasedOutputDimensions() throws {
        let modelRoot = try makeValidLingBotModelRoot()
        let output = makeTempOutput(name: "refiner-defaults.mp4")
        let cmd = try VideoGenerate.parse([
            "a robot folds a towel",
            "--model-root", modelRoot.path,
            "--variant", "lingbot",
            "--refiner",
            "--output", output.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )
        XCTAssertEqual(envelope.result.plan.refinerWidth, 1_920)
        XCTAssertEqual(envelope.result.plan.refinerHeight, 1_088)
    }

    func testLingBotPreflightUsesSixteenPixelAndFourFrameStrides() throws {
        let modelRoot = try makeValidLingBotModelRoot()
        let output = makeTempOutput(name: "lingbot.mp4")
        let cmd = try VideoGenerate.parse([
            "a robot folds a towel",
            "--model", "video-lingbot-dense-1.3b",
            "--model-root", modelRoot.path,
            "--output", output.path,
            "--width", "333",
            "--height", "201",
            "--num-frames", "10",
            "--steps", "12",
            "--guidance-scale", "2.5",
            "--shift", "3.5",
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.model.layout, "lingbot_dense")
        XCTAssertEqual(envelope.result.plan.variant, "lingbot")
        XCTAssertEqual(envelope.result.plan.resolvedWidth, 320)
        XCTAssertEqual(envelope.result.plan.resolvedHeight, 192)
        XCTAssertEqual(envelope.result.plan.resolvedNumFrames, 9)
        XCTAssertEqual(envelope.result.plan.steps, 12)
        XCTAssertEqual(envelope.result.plan.guidanceScale, 2.5)
        XCTAssertEqual(envelope.result.plan.shift, 3.5)
        XCTAssertFalse(envelope.result.plan.writesAudio)
        let actionArguments = envelope.actions
            .first { $0.id == "start-video-generation" }?
            .command?.argv
        let variantIndex = try XCTUnwrap(actionArguments?.firstIndex(of: "--variant"))
        XCTAssertEqual(actionArguments?[variantIndex + 1], "lingbot")
    }

    func testVideoGeneratePreflightReportsResolvedPlan() throws {
        let modelRoot = try makeValidLTXModelRoot()
        let sourceImage = try makeTempFile(name: "start.png")
        let endImage = try makeTempFile(name: "end.png")
        let output = makeTempOutput(name: "clip.mp4")
        let cmd = try VideoGenerate.parse([
            "a flower opens from bud to bloom",
            "--model-root", modelRoot.path,
            "--output", output.path,
            "--width", "1025",
            "--height", "578",
            "--num-frames", "66",
            "--image", sourceImage.path,
            "--end-image", endImage.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.command, ["video", "generate"])
        XCTAssertEqual(envelope.mode, .preflight)
        XCTAssertEqual(envelope.result.model.kind, "model_root")
        XCTAssertTrue(envelope.result.model.installed)
        XCTAssertEqual(envelope.result.output.path, output.path)
        XCTAssertEqual(envelope.result.inputs.mode, "directed_image_to_video")
        XCTAssertEqual(envelope.result.plan.resolvedWidth, 1024)
        XCTAssertEqual(envelope.result.plan.resolvedHeight, 576)
        XCTAssertEqual(envelope.result.plan.resolvedNumFrames, 65)
        XCTAssertEqual(envelope.result.plan.seed, 42)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "dimensions_will_be_adjusted" })
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "num_frames_will_be_adjusted" })
        XCTAssertTrue(envelope.actions.contains { $0.id == "start-video-generation" && $0.enabled })

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(VideoGenerationPreflightEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded, envelope)
    }

    func testVideoGeneratePreflightBlocksMissingSourceImage() throws {
        let modelRoot = try makeValidLTXModelRoot()
        let output = makeTempOutput(name: "clip.mp4")
        let missingImage = makeTempOutput(name: "missing.png")
        let cmd = try VideoGenerate.parse([
            "woman walking in neon rain",
            "--model-root", modelRoot.path,
            "--output", output.path,
            "--image", missingImage.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "source_image_missing" })
        XCTAssertTrue(envelope.actions.contains { $0.id == "start-video-generation" && !$0.enabled })
    }

    func testVideoGeneratePreflightBlocksEndImageWithoutSourceImage() throws {
        let modelRoot = try makeValidLTXModelRoot()
        let endImage = try makeTempFile(name: "end.png")
        let output = makeTempOutput(name: "clip.mp4")
        let cmd = try VideoGenerate.parse([
            "a car drives from dawn into sunset",
            "--model-root", modelRoot.path,
            "--output", output.path,
            "--end-image", endImage.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "end_image_requires_source_image" })
    }

    func testVideoGeneratePreflightWarnsForUnifiedAVNon24FPS() throws {
        let modelRoot = try makeValidLTXModelRoot()
        let output = makeTempOutput(name: "clip.mp4")
        let cmd = try VideoGenerate.parse([
            "dialogue with background music",
            "--variant", "unified-av",
            "--fps", "8",
            "--model-root", modelRoot.path,
            "--output", output.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .warning)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "unified_av_fps_unusual" })
        XCTAssertTrue(envelope.actions.contains { $0.id == "start-video-generation" && $0.enabled })
    }

    func testVideoExportLatentsParsesOverrides() throws {
        let cmd = try VideoExportLatents.parse([
            "a neon city skyline",
            "--model-root", "/tmp/ltx",
            "--output", "/tmp/out.safetensors",
            "--width", "1024",
            "--height", "576",
            "--num-frames", "33",
            "--seed", "7",
        ])

        XCTAssertEqual(cmd.prompt, "a neon city skyline")
        XCTAssertEqual(cmd.modelRoot, "/tmp/ltx")
        XCTAssertEqual(cmd.output, "/tmp/out.safetensors")
        XCTAssertEqual(cmd.width, 1024)
        XCTAssertEqual(cmd.height, 576)
        XCTAssertEqual(cmd.numFrames, 33)
        XCTAssertEqual(cmd.seed, 7)
    }

    func testValidateNativeModelRootRejectsMissingTokenizerDirectory() throws {
        let rootURL = try makeTempDirectory()
        try createFile(rootURL.appendingPathComponent("text_encoder/config.json"))
        try createFile(rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        try createFile(rootURL.appendingPathComponent("ltx-2-19b-distilled.safetensors"))
        try createFile(rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors"))

        XCTAssertThrowsError(try validateNativeModelRoot(rootURL))
    }

    func testValidateNativeModelRootAcceptsExpectedLayout() throws {
        let rootURL = try makeTempDirectory()
        try createFile(rootURL.appendingPathComponent("text_encoder/config.json"))
        try createFile(rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        try FileManager.default.createDirectory(
            at: rootURL.appendingPathComponent("tokenizer", isDirectory: true),
            withIntermediateDirectories: true
        )
        try createFile(rootURL.appendingPathComponent("ltx-2-19b-distilled.safetensors"))
        try createFile(rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors"))

        XCTAssertNoThrow(try validateNativeModelRoot(rootURL))
    }

    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VideoCommandTests.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        temporaryDirectories.append(directory)
        return directory
    }

    private func createFile(_ url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = Data("fixture".utf8)
        try data.write(to: url)
    }

    private func makeValidLTXModelRoot() throws -> URL {
        let rootURL = try makeTempDirectory()
        try createFile(rootURL.appendingPathComponent("text_encoder/config.json"))
        try createFile(rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        try FileManager.default.createDirectory(
            at: rootURL.appendingPathComponent("tokenizer", isDirectory: true),
            withIntermediateDirectories: true
        )
        try createFile(rootURL.appendingPathComponent("ltx-2-19b-distilled.safetensors"))
        try createFile(rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors"))
        return rootURL
    }

    private func makeValidLingBotModelRoot() throws -> URL {
        let root = try makeTempDirectory()
        for directory in ["processor", "text_encoder", "transformer", "vae"] {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(directory, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
        try Data("{}".utf8).write(to: root.appendingPathComponent("model_index.json"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("processor/tokenizer.json"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("processor/tokenizer_config.json"))
        try Data(Self.lingBotTextConfig.utf8).write(to: root.appendingPathComponent("text_encoder/config.json"))
        try Data("{\"metadata\":{},\"weight_map\":{}}".utf8)
            .write(to: root.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        try Data(Self.lingBotTransformerConfig.utf8).write(to: root.appendingPathComponent("transformer/config.json"))
        try Data().write(to: root.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors"))
        try Data(Self.lingBotVAEConfig.utf8).write(to: root.appendingPathComponent("vae/config.json"))
        try Data().write(to: root.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        return root
    }

    private static let lingBotTransformerConfig = """
    {"axes_dims":[32,48,48],"axes_lens":[8192,1024,1024],"depth":24,"freq_dim":256,"hidden_size":2048,"in_channels":16,"intermediate_size":6144,"norm_eps":0.000001,"num_attention_heads":16,"num_experts":0,"out_bias":true,"out_channels":16,"patch_embed_bias":true,"patch_size":[1,2,2],"qkv_bias":false,"rope_theta":256.0,"text_dim":2560,"timestep_mlp_bias":true}
    """

    private static let lingBotTextConfig = """
    {"text_config":{"head_dim":128,"hidden_size":2560,"intermediate_size":9728,"max_position_embeddings":262144,"num_attention_heads":32,"num_hidden_layers":36,"num_key_value_heads":8,"rms_norm_eps":0.000001,"rope_scaling":{"mrope_interleaved":true,"mrope_section":[24,20,20]},"rope_theta":5000000,"vocab_size":151936}}
    """

    private static let lingBotVAEConfig = """
    {"base_dim":96,"dim_mult":[1,2,4,4],"latents_mean":[-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921],"latents_std":[2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.916],"num_res_blocks":2,"temperal_downsample":[false,true,true],"z_dim":16}
    """

    private func makeTempFile(name: String) throws -> URL {
        let directory = try makeTempDirectory()
        let url = directory.appendingPathComponent(name)
        try createFile(url)
        return url
    }

    private func makeTempOutput(name: String) -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VideoCommandTests.\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        temporaryDirectories.append(directory)
        return directory.appendingPathComponent(name)
    }
}
