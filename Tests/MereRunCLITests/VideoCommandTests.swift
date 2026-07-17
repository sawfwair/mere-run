import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class VideoCommandTests: XCTestCase {
    private var temporaryDirectories: [URL] = []

    override func tearDown() {
        for directory in temporaryDirectories {
            try? FileManager.default.removeItem(at: directory)
        }
        temporaryDirectories.removeAll()
        super.tearDown()
    }

    func testVideoCommandExposesGenerateExportLatentsAndSession() {
        let commandNames = Set(Video.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(commandNames, Set(["generate", "export-latents", "session"]))
    }

    func testVideoGenerateParsesDefaults() throws {
        let cmd = try VideoGenerate.parse([
            "a cinematic drone flythrough"
        ])

        XCTAssertEqual(cmd.prompt, "a cinematic drone flythrough")
        XCTAssertEqual(cmd.variant, .distilled)
        XCTAssertEqual(cmd.width, 768)
        XCTAssertEqual(cmd.height, 512)
        XCTAssertEqual(cmd.numFrames, 65)
        XCTAssertNil(cmd.duration)
        XCTAssertEqual(cmd.fps, 24)
        XCTAssertEqual(cmd.imageStrength, 1.0)
        XCTAssertNil(cmd.endImage)
        XCTAssertEqual(cmd.endImageStrength, 1.0)
        XCTAssertNil(cmd.modelRoot)
        XCTAssertFalse(cmd.preflight)
        XCTAssertFalse(cmd.json)
        XCTAssertFalse(cmd.timings)
        XCTAssertNil(cmd.timingsOutput)
    }

    func testVideoGenerateParsesTimingOptions() throws {
        let cmd = try VideoGenerate.parse([
            "a cinematic drone flythrough",
            "--timings",
            "--timings-output", "/tmp/ltx-timings.json",
        ])

        XCTAssertTrue(cmd.timings)
        XCTAssertEqual(cmd.timingsOutput, "/tmp/ltx-timings.json")
    }

    func testVideoGenerateRejectsTimingOptionsForUnsupportedLane() async throws {
        let cmd = try VideoGenerate.parse([
            "a cinematic drone flythrough",
            "--timings",
        ])

        do {
            try await cmd.run()
            XCTFail("Expected timing options without unified AV or A2Vid to fail validation.")
        } catch {
            XCTAssertTrue(String(describing: error).contains("require --variant unified-av or --audio"))
        }
    }

    func testVideoSessionParsesDefaults() throws {
        let cmd = try VideoSession.parse([])

        XCTAssertEqual(cmd.model, ModelResolver.ModelID.ltxVideo23AVMLX.rawValue)
        XCTAssertNil(cmd.modelRoot)
        XCTAssertFalse(cmd.quiet)
    }

    func testVideoSessionRequestDecodesSnakeCase() throws {
        let data = Data(
            """
            {
              "id": "fox-1",
              "prompt": "a fox runs across snow",
              "output": "/tmp/fox.mp4",
              "width": 512,
              "height": 320,
              "num_frames": 33,
              "fps": 24,
              "seed": 7,
              "image_strength": 0.8,
              "end_image_strength": 0.7
            }
            """.utf8
        )
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let request = try decoder.decode(LTXVideoSessionRequest.self, from: data)

        XCTAssertEqual(request.id, "fox-1")
        XCTAssertEqual(request.prompt, "a fox runs across snow")
        XCTAssertEqual(request.output, "/tmp/fox.mp4")
        XCTAssertEqual(request.width, 512)
        XCTAssertEqual(request.height, 320)
        XCTAssertEqual(request.numFrames, 33)
        XCTAssertEqual(request.imageStrength ?? 0, 0.8, accuracy: 0.0001)
        XCTAssertEqual(request.endImageStrength ?? 0, 0.7, accuracy: 0.0001)
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
        XCTAssertEqual(cmd.resolvedRequestedModel, ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)
    }

    func testVideoGenerateAudioSelectsNativeA2VidDefaults() throws {
        let cmd = try VideoGenerate.parse([
            "the singer performs beneath sweeping blue spotlights",
            "--audio", "/tmp/song.wav",
            "--audio-start-time", "42",
            "--a2v-guidance-scale", "2.5",
            "--video-cfg-guidance-scale", "3.5",
            "--audio-cfg-guidance-scale", "6.5",
            "--v2a-guidance-scale", "2.75",
            "--a2v-steps", "28",
        ])

        XCTAssertEqual(cmd.audio, "/tmp/song.wav")
        XCTAssertEqual(cmd.audioStartTime, 42)
        XCTAssertEqual(cmd.a2vGuidanceScale, 2.5, accuracy: 0.0001)
        XCTAssertEqual(cmd.videoCFGGuidanceScale, 3.5, accuracy: 0.0001)
        XCTAssertEqual(cmd.audioCFGGuidanceScale, 6.5, accuracy: 0.0001)
        XCTAssertEqual(cmd.v2aGuidanceScale, 2.75, accuracy: 0.0001)
        XCTAssertEqual(cmd.a2vSteps, 28)
        XCTAssertEqual(cmd.resolvedRequestedModel, ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)
    }

    func testVideoGenerateA2VidPreflightReportsSourceAudioContract() throws {
        let modelRoot = try makeValidA2VidModelRoot()
        let sourceAudio = try makeTempFile(name: "song.wav")
        let sourceImage = try makeTempFile(name: "artist.png")
        let output = makeTempOutput(name: "shot.mp4")
        let cmd = try VideoGenerate.parse([
            "the singer performs beneath sweeping blue spotlights",
            "--model-root", modelRoot.path,
            "--audio", sourceAudio.path,
            "--audio-start-time", "42",
            "--duration", "5",
            "--image", sourceImage.path,
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
        XCTAssertEqual(envelope.result.model.layout, "ltx23_a2vid_split")
        XCTAssertEqual(envelope.result.inputs.mode, "audio_and_image_to_video")
        XCTAssertEqual(envelope.result.inputs.sourceAudio?.path, sourceAudio.path)
        XCTAssertEqual(envelope.result.plan.variant, "audio-to-video")
        XCTAssertEqual(envelope.result.plan.resolvedNumFrames, 121)
        XCTAssertEqual(envelope.result.plan.resolvedDurationSeconds ?? 0, 121.0 / 24.0, accuracy: 0.0001)
        XCTAssertEqual(envelope.result.plan.resolvedAudioStartTime, 42)
        XCTAssertTrue(envelope.result.plan.audioConditioning)
        XCTAssertTrue(envelope.result.plan.preservesSourceAudio)
        XCTAssertTrue(envelope.result.plan.writesAudio)
        XCTAssertTrue(envelope.actions.contains { $0.id == "reveal-source-audio" && $0.enabled })
    }

    func testVideoGenerateFullModelPreflightReportsUnifiedBundle() throws {
        let modelRoot = try makeValidFullModelRoot()
        let output = makeTempOutput(name: "clip.mp4")
        let cmd = try VideoGenerate.parse([
            "dialogue with synchronized ambience",
            "--variant", "unified-av",
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

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.model.layout, "ltx23_full_split")
        XCTAssertEqual(envelope.result.plan.variant, "unified-av")
        XCTAssertTrue(envelope.result.plan.writesAudio)
    }

    func testVideoGenerateA2VidPreflightBlocksIncompatibleManagedModel() throws {
        let sourceAudio = try makeTempFile(name: "song.wav")
        let output = makeTempOutput(name: "shot.mp4")
        let cmd = try VideoGenerate.parse([
            "a kinetic live performance",
            "--model", ModelResolver.ModelID.wan22TI2V5BMLX.rawValue,
            "--audio", sourceAudio.path,
            "--output", output.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "audio_model_incompatible" })
    }

    func testNearestLTXFrameCountUsesClosestLegalFrameCount() {
        XCTAssertEqual(nearestLTXFrameCount(duration: 15, fps: 24), 361)
        XCTAssertEqual(nearestLTXFrameCount(duration: 5, fps: 24), 121)
        XCTAssertEqual(nearestLTXFrameCount(duration: 15, fps: 8), 121)
    }

    func testNearestWanFrameCountUsesFourFrameLatentCadence() {
        XCTAssertEqual(nearestWanFrameCount(duration: 1, fps: 24), 25)
        XCTAssertEqual(nearestWanFrameCount(duration: 0.5, fps: 24), 13)
        XCTAssertEqual(nearestWanFrameCount(duration: 0.1, fps: 24), 5)
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

    func testValidateNativeModelRootAcceptsFullLTX23Layout() throws {
        XCTAssertNoThrow(try validateNativeModelRoot(makeValidFullModelRoot()))
    }

    func testWanPreflightUsesNativeSpatialAndTemporalGeometry() throws {
        let modelRoot = try makeValidWanModelRoot()
        let sourceImage = try makeTempFile(name: "start.png")
        let output = makeTempOutput(name: "clip.mp4")
        let cmd = try VideoGenerate.parse([
            "the camera walks forward",
            "--model-root", modelRoot.path,
            "--output", output.path,
            "--width", "1057",
            "--height", "577",
            "--num-frames", "18",
            "--image", sourceImage.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.model.layout, "wan22_ti2v_mlx")
        XCTAssertEqual(envelope.result.plan.variant, "wan22-ti2v")
        XCTAssertEqual(envelope.result.plan.resolvedWidth, 1_056)
        XCTAssertEqual(envelope.result.plan.resolvedHeight, 576)
        XCTAssertEqual(envelope.result.plan.resolvedNumFrames, 17)
        XCTAssertFalse(envelope.result.plan.writesAudio)
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

    private func makeValidA2VidModelRoot() throws -> URL {
        let rootURL = try makeTempDirectory()
        for name in [
            "split_model.json",
            "config.json",
            "connector.safetensors",
            "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
        ] {
            try createFile(rootURL.appendingPathComponent(name))
        }
        return rootURL
    }

    private func makeValidFullModelRoot() throws -> URL {
        let rootURL = try makeValidA2VidModelRoot()
        try createFile(rootURL.appendingPathComponent("vocoder.safetensors"))
        return rootURL
    }

    private func makeValidWanModelRoot() throws -> URL {
        let rootURL = try makeTempDirectory()
        let config = """
        {
          "model_type": "ti2v",
          "model_version": "2.2",
          "patch_size": [1, 2, 2],
          "text_len": 512,
          "in_dim": 48,
          "dim": 3072,
          "ffn_dim": 14336,
          "text_dim": 4096,
          "out_dim": 48,
          "num_heads": 24,
          "num_layers": 30,
          "vae_stride": [4, 16, 16],
          "vae_z_dim": 48,
          "sample_shift": 5.0,
          "sample_steps": 40,
          "sample_guide_scale": 5.0,
          "sample_fps": 24,
          "frame_num": 81,
          "max_area": 901120
        }
        """
        try Data(config.utf8).write(to: rootURL.appendingPathComponent("config.json"))
        for name in ["model.safetensors", "t5_encoder.safetensors", "tokenizer.json", "vae.safetensors"] {
            try createFile(rootURL.appendingPathComponent(name))
        }
        return rootURL
    }

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
