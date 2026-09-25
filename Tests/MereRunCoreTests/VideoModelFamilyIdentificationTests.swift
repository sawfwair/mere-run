import Foundation
import MereRunContract
import Testing

@testable import MereRunCore

/// The video commands' own routing and the contract resolve every managed id and every local
/// layout to the same family.
@Suite struct VideoModelFamilyIdentificationTests {
    private let fastH3 = ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue

    @Test func coreProfilesAgreeWithTheContractForEveryManagedVideoModel() throws {
        let routing = try #require(MereRunCapabilityCatalog.videoGenerate.routing)
        for family in routing.families {
            for model in family.models {
                let profile = VideoGenerationModelProfile.managed(model)
                #expect(profile != .unknown, "\(model)")
                // The FastH3 id with an adapter runs FL2VA's layout; only the embedded one is FastH3.
                let expected = family.id == "h3-fast-adapter" ? "h3-fl2va" : family.id
                #expect(profile.videoGenerateFamily(fastH3: family.id == "h3-fast") == expected, "\(model)")
            }
        }
        for excluded in routing.excludedModels {
            #expect(VideoGenerationModelProfile.managed(excluded.id) == .unknown, "\(excluded.id)")
        }
        // The install-dependent ids are the contract's identified models; before resolution the
        // Full and A2Vid ids check as their own layout and the merged id is left unchecked.
        #expect(Set(VideoGenerationModelProfile.installDependentLayouts.keys) == Set(routing.identifiedModels))
        #expect(VideoGenerationModelProfile.managed("video-ltx23-full-mlx") == .ltx23Full)
        #expect(VideoGenerationModelProfile.managed("video-ltx23-a2vid-mlx") == .ltx23AudioToVideo)
        #expect(VideoGenerationModelProfile.managed("video-ltx-av") == .unknown)
    }

    @Test func theGenerateProbeReportsTheLayoutTheOperationObserves() throws {
        let cases: [(URL, String)] = try [
            (makeRoot(Self.ltx23SplitFiles, config: #"{"model_version":"2.3"}"#), "ltx23-distilled"),
            (makeRoot(Self.ltx23A2VidFiles), "ltx23-a2vid"),
            (makeRoot(Self.ltx23A2VidFiles + ["vocoder.safetensors"]), "ltx23-full"),
            (makeRoot(LTX25Resources.requiredRelativePaths + [LTX25Resources.textEncoderRelativePath]), "ltx25-distilled"),
            (makeRoot(LTX25Resources.fullRequiredRelativePaths), "ltx25-full"),
            (makeRoot(Self.wanFiles, config: Self.wanConfig), "wan22-ti2v"),
            (makeRoot(MiniMaxH3Resources.requiredFiles, config: Self.h3Config(task: "fl2va")), "h3-fl2va"),
            (makeRoot(MiniMaxH3Resources.requiredFiles, config: Self.h3Config(task: "ref2va")), "h3-ref2va"),
            (makeRoot(["notes.txt"]), "ltx-merged")
        ]
        for (root, family) in cases {
            defer { try? FileManager.default.removeItem(at: root) }
            #expect(VideoGenerationModelProfile.observe(root: root).videoGenerateFamily(fastH3: false) == family)
            for arguments in [["--model-root", root.path], ["--model", root.path]] {
                #expect(identify("video.generate", root.path, arguments) == family, "\(arguments)")
            }
        }
        #expect(identify("video.generate", "/nonexistent/video-root", ["--model", "/nonexistent/video-root"]) == nil)
    }

    /// FastH3 is laid out like FL2VA; its managed id picks the embedded adapter, even when
    /// `--model-root` names the folder, unless `--h3-adapter` replaces it.
    @Test func aFastH3FolderRunsAsFastH3OnlyWithTheFastH3Id() throws {
        let root = try makeRoot(MiniMaxH3Resources.requiredFiles, config: Self.h3Config(task: "fl2va"))
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(identify("video.generate", root.path, ["--model-root", root.path, "--model", fastH3]) == "h3-fast")
        #expect(identify(
            "video.generate", root.path, ["--model-root", root.path, "--model", fastH3, "--h3-adapter", "turbo.safetensors"]
        ) == "h3-fl2va")
        #expect(identify("video.generate", root.path, ["--model-root", root.path]) == "h3-fl2va")
        #expect(identify("video.generate", root.path, ["--model", root.path]) == "h3-fl2va")
    }

    @Test func retakeAndSessionProbesFollowTheirCommandsLayoutChecks() throws {
        let split = try makeRoot(Self.ltx23SplitFiles, config: #"{"model_version":"2.3"}"#)
        let full23 = try makeRoot(Self.ltx23A2VidFiles + ["vocoder.safetensors"])
        let a2vid = try makeRoot(Self.ltx23A2VidFiles)
        let distilled25 = try makeRoot(LTX25Resources.requiredRelativePaths + [LTX25Resources.textEncoderRelativePath])
        let full25 = try makeRoot(LTX25Resources.fullRequiredRelativePaths)
        defer { for root in [split, full23, a2vid, distilled25, full25] { try? FileManager.default.removeItem(at: root) } }

        #expect(identify("video.retake", full25.path) == "ltx25-full")
        #expect(identify("video.retake", distilled25.path) == "ltx25-distilled")
        #expect(identify("video.retake", full23.path) == nil)

        #expect(identify("video.session", full25.path) == "ltx25-full")
        #expect(identify("video.session", distilled25.path) == "ltx25-distilled")
        #expect(identify("video.session", full23.path) == "ltx23-full")
        #expect(identify("video.session", split.path) == "ltx23-distilled")
        #expect(identify("video.session", a2vid.path) == nil, "an A2Vid folder without a vocoder can't hold a session")
    }

    // MARK: - Fixtures

    private func identify(_ capabilityID: String, _ model: String, _ arguments: [String]? = nil) -> String? {
        let capability = MereRunCapabilityCatalog.command(id: capabilityID)!
        let invocation = MereRunCommandInvocation(capability: capability, arguments: arguments ?? ["--model-root", model])
        guard case .family(let family)? = ModelFamilyIdentifier.identify(
            capabilityID: capabilityID, model: model, invocation: invocation
        ) else { return nil }
        return family
    }

    private func makeRoot(_ files: [String], config: String? = nil) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(path: "video-family-\(UUID().uuidString)")
        for path in files {
            let file = root.appending(path: path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: file)
        }
        if let config {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            try Data(config.utf8).write(to: root.appending(path: "config.json"))
        }
        return root
    }

    private static let ltx23SplitFiles = [
        "split_model.json", "transformer-distilled.safetensors", "vae_decoder.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors"
    ]

    private static let ltx23A2VidFiles = [
        "split_model.json", "config.json", "connector.safetensors", "transformer-dev.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
        "audio_vae.safetensors", "spatial_upscaler_x2_v1_1.safetensors"
    ]

    private static let wanFiles = ["model.safetensors", "t5_encoder.safetensors", "tokenizer.json", "vae.safetensors"]

    private static let wanConfig = """
    {"model_type": "ti2v", "model_version": "2.2", "patch_size": [1, 2, 2], "text_len": 512, "in_dim": 48,
     "dim": 3072, "ffn_dim": 14336, "text_dim": 4096, "out_dim": 48, "num_heads": 24, "num_layers": 30,
     "vae_stride": [4, 16, 16], "vae_z_dim": 48, "sample_shift": 5.0, "sample_steps": 40,
     "sample_guide_scale": 5.0, "sample_fps": 24, "frame_num": 81, "max_area": 901120}
    """

    private static func h3Config(task: String) -> String {
        """
        {"model_type": "minimax_h3", "partition": "\(task)",
         "transformer": {"hidden_size": 5376, "num_layers": 50, "num_attention_heads": 56, "attention_head_dim": 128,
                         "ffn_hidden_size": 14336, "latents_dim": 24, "audio_latents_dim": 32, "text_dim": 5120,
                         "time_embed_dim": 2688, "rope_inv_freq_len": 64},
         "sigma_shift_scales": {"video": 5.0, "audio": 5.0}}
        """
    }
}
