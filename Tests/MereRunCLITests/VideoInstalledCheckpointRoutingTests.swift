import Foundation
import MereRunContract
import MereRunCore
import XCTest

@testable import MereRunCLI

/// The LTX 2.3 ids whose checkpoint depends on what is installed resolve at the gate to the
/// folder `VideoGenerationModelResolver` will run, and to the layout they name when nothing is
/// installed. Each test points the model store at its own fixture folders, which is process
/// state, so these run serially like `CapabilityGateRootValidationTests`.
final class VideoInstalledCheckpointRoutingTests: XCTestCase {
    private var modelsRoot: URL!
    private var originalLTXRoot: String?

    override func setUpWithError() throws {
        try super.setUpWithError()
        modelsRoot = FileManager.default.temporaryDirectory.appending(path: "video-installed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelsRoot, withIntermediateDirectories: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)
        originalLTXRoot = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_MODEL_ROOT"]
        unsetenv("MERERUN_VIDEO_LTX_MODEL_ROOT")
    }

    override func tearDown() {
        MereRunModelPaths.setProcessModelsDirOverride(nil)
        if let originalLTXRoot { setenv("MERERUN_VIDEO_LTX_MODEL_ROOT", originalLTXRoot, 1) }
        try? FileManager.default.removeItem(at: modelsRoot)
        super.tearDown()
    }

    private func report(_ commandLine: [String]) throws -> MereRunFamilyResolutionReport {
        try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: commandLine)).report
    }

    /// An LTX 2.3 Full (or, without the vocoder, A2Vid) folder installed under `id`.
    @discardableResult
    private func install(_ id: ModelResolver.ModelID, vocoder: Bool) throws -> URL {
        let root = modelsRoot.appending(path: id.rawValue)
        let files = [
            "split_model.json", "config.json", "connector.safetensors", "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
            "audio_vae.safetensors", "spatial_upscaler_x2_v1_1.safetensors", "embedded_config.json",
            "spatial_upscaler_x2_v1_1_config.json"
        ] + (vocoder ? ["vocoder.safetensors"] : [])
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for file in files {
            try Data().write(to: root.appending(path: file))
        }
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        return root
    }

    private func resolvedRoot(_ model: String, variant: LTXVideoVariant) async throws -> URL {
        try await VideoGenerationModelResolver.resolve(
            explicitModelRoot: nil, requestedModel: model, variant: variant, allowAutoDownload: false
        )
    }

    func testWithNothingInstalledEachIdRunsTheLayoutItNames() throws {
        let merged = try report(["video", "generate", "a", "--model", "video-ltx-av", "--audio", "song.wav"])
        XCTAssertEqual(merged.family, "ltx-merged")
        XCTAssertEqual(merged.violations, [
            "--audio is not supported by LTX (merged). It applies to LTX-2.3 Full, LTX-2.3 A2Vid and LTX-2.5 Full."
        ])
        XCTAssertEqual(try report(["video", "generate", "a", "--model", "video-ltx23-full-mlx"]).family, "ltx23-full")
        XCTAssertEqual(try report(["video", "generate", "a", "--model", "video-ltx23-a2vid-mlx"]).family, "ltx23-a2vid")
        let finalDefault = try report(["video", "generate", "a", "--quality", "final"])
        XCTAssertEqual(finalDefault.family, "ltx23-full")
        XCTAssertEqual(finalDefault.model, "video-ltx23-full-mlx")
        XCTAssertEqual(finalDefault.source, .defaultModel)

        XCTAssertEqual(try report(["video", "session", "--model", "video-ltx23-full-mlx", "--ltx-teacache"]).warnings.count, 1)
        XCTAssertEqual(try report(["video", "session", "--model", "video-ltx-av"]).source, .unidentified)
        XCTAssertEqual(try report(["video", "retake", "a", "--model", "video-ltx-av"]).source, .unidentified)
    }

    /// `video-ltx-av` with audio-video output takes the first suggested folder, an installed
    /// LTX 2.3 Full, so source audio passes the gate as it runs; video-only output does not look
    /// there.
    func testTheMergedIdRunsASuggestedLTX23FullFolderForAudioVideo() async throws {
        let full = try install(.ltxVideo23FullMLX, vocoder: true)
        let resolved = try await resolvedRoot("video-ltx-av", variant: .unifiedAV)
        XCTAssertEqual(resolved.standardizedFileURL, full.standardizedFileURL)
        for extra in [["--audio", "song.wav"], ["--output-mode", "audio-video"], ["--variant", "unified-av"]] {
            let result = try report(["video", "generate", "a", "--model", "video-ltx-av"] + extra)
            XCTAssertEqual(result.family, "ltx23-full", "\(extra)")
            XCTAssertEqual(result.violations, [], "\(extra)")
        }
        XCTAssertEqual(try report(["video", "generate", "a", "--model", "video-ltx-av"]).family, "ltx-merged")
        XCTAssertEqual(try report(["video", "session", "--model", "video-ltx-av"]).family, "ltx23-full")
    }

    /// The LTX 2.3 Full id falls back to an installed A2Vid folder, which can't hold a session.
    func testTheFullIdRunsAnInstalledA2VidFolder() async throws {
        let a2vid = try install(.ltxVideo23A2VMLX, vocoder: false)
        let resolved = try await resolvedRoot("video-ltx23-full-mlx", variant: .unifiedAV)
        XCTAssertEqual(resolved.standardizedFileURL, a2vid.standardizedFileURL)
        let generate = try report(["video", "generate", "a", "--model", "video-ltx23-full-mlx"])
        XCTAssertEqual(generate.family, "ltx23-a2vid")
        XCTAssertEqual(generate.source, .identified)
        XCTAssertEqual(try report(["video", "generate", "a", "--quality", "final"]).family, "ltx23-a2vid")
        XCTAssertEqual(try report(["video", "session", "--model", "video-ltx23-full-mlx"]).source, .unidentified)
    }

    /// `MERERUN_VIDEO_LTX_MODEL_ROOT` comes first among the suggested folders; retake runs the
    /// merged id when it points at an LTX 2.5 folder.
    func testTheSuggestedRootVariableCanSendTheMergedIdToLTX25() throws {
        let root = modelsRoot.appending(path: "custom-ltx25")
        for path in LTX25Resources.fullRequiredRelativePaths {
            let file = root.appending(path: path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: file)
        }
        setenv("MERERUN_VIDEO_LTX_MODEL_ROOT", root.path, 1)
        let retake = try report(["video", "retake", "a", "--model", "video-ltx-av", "--steps", "30"])
        XCTAssertEqual(retake.family, "ltx25-full")
        XCTAssertEqual(retake.violations, [])
        XCTAssertEqual(try report(["video", "generate", "a", "--model", "video-ltx-av", "--dfr"]).family, "ltx25-full")
    }
}
