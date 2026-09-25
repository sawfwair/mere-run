import Foundation
import MereRunContract
import MereRunCore
import Testing

@testable import MereRunCLI

/// Routing the capability gate can't derive from the contract alone: `video generate`'s
/// option-driven defaults, `--model-root` precedence, FastH3, `world serve`'s backend, and the
/// scopes on the flags that also pick a default model, which the generated cases skip.
@Suite struct VideoCapabilityRoutingTests {
    private func report(_ commandLine: [String]) throws -> MereRunFamilyResolutionReport {
        try #require(CLICapabilityGate.evaluate(commandLine: commandLine)).report
    }

    /// Every default rule's every condition selects the same checkpoint in the contract and in
    /// `VideoGenerationOptions.resolvedRequestedModel`, in the same order.
    @Test func defaultModelRulesMatchTheCommandsOwnDefault() throws {
        let capability = MereRunCapabilityCatalog.videoGenerate
        let routing = try #require(capability.routing)
        var lines: [([String], String)] = [([], "video-ltx23-av-mlx")]
        for rule in routing.defaultModels {
            let model = try #require(rule.models.first)
            for condition in rule.whenAny {
                for value in condition.values ?? [sampleValue(for: condition.flag)] {
                    lines.append((tokens(condition.flag, value), model))
                }
            }
        }
        lines += [
            (["--dfr", "--hdr", "srgb-linear"], "video-ltx25-full-bf16"),
            (["--hdr", "acescg", "--audio", "a.wav"], "video-ltx25-distilled-bf16"),
            (["--num-generated-keyframes", "0"], "video-ltx23-av-mlx"),
            (["--ltx-pipeline", "two-stage", "--ltx-preset", "standard"], "video-ltx23-av-mlx"),
            (["--quality", "draft"], "video-ltx23-av-mlx"),
            (["--variant", "distilled"], "video-ltx23-av-mlx")
        ]
        for (arguments, model) in lines {
            let command = try VideoGenerate.parse(["a prompt"] + arguments)
            #expect(command.resolvedRequestedModel == model, "\(arguments)")
            let resolved = try report(["video", "generate", "a prompt"] + arguments)
            #expect(resolved.model == model && resolved.source == .defaultModel, "\(arguments): \(resolved)")
        }
    }

    @Test func aModelRootWinsOverTheModelAndIsIdentifiedByItsFiles() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "video-routing-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        for path in LTX25Resources.fullRequiredRelativePaths {
            let file = root.appending(path: path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: file)
        }
        let resolved = try report([
            "video", "generate", "a", "--model", "video-wan22-ti2v-5b-mlx", "--model-root", root.path, "--dfr"
        ])
        #expect(resolved.family == "ltx25-full" && resolved.source == .identified && resolved.violations.isEmpty)
        let command = try VideoGenerate.parse(["a", "--model", "video-wan22-ti2v-5b-mlx", "--model-root", root.path])
        #expect(command.makeGenerationOptions(outputURL: root.appending(path: "a.mp4")).observedProfile() == .ltx25Full)

        let retake = try report(["video", "retake", "a", "--model-root", root.path, "--model", "video-ltx-av"])
        #expect(retake.family == "ltx25-full" && retake.violations.isEmpty, "\(retake)")
        let session = try report(["video", "session", "--model-root", root.path])
        #expect(session.family == "ltx25-full" && session.violations.isEmpty, "\(session)")
    }

    @Test func fastH3IsTheManagedIdWithItsEmbeddedAdapter() throws {
        let fastH3 = ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
        let options = try VideoGenerate.parse(["a", "--model", fastH3])
            .makeGenerationOptions(outputURL: URL(fileURLWithPath: "/tmp/a.mp4"))
        #expect(options.usesEmbeddedFastH3Adapter && options.observedProfile().isH3)
        #expect(try report(["video", "generate", "a", "--model", fastH3]).family == "h3-fast")

        let base = ["video", "generate", "a", "--model", fastH3]
        #expect(try report(base + ["--steps", "5", "--h3-acceleration", "quality"]).violations.isEmpty)
        #expect(try report(base + ["--steps", "9"]).violations
            == ["--steps 9 is not supported by MiniMax-H3 FastH3; it runs 5. Remove --steps or pass 5."])
        #expect(try report(base + ["--image", "a.png"]).violations
            == ["--image is not supported by MiniMax-H3 FastH3. It applies to LTX (merged), LTX-2.3 Distilled, "
                + "LTX-2.3 Full, LTX-2.3 A2Vid, LTX-2.5 Distilled, LTX-2.5 Distilled with the diffusion decoder, LTX-2.5 Full, "
                + "Wan 2.2 TI2V, MiniMax-H3 FL2VA, "
                + "MiniMax-H3 FL2VA 4-bit and MiniMax-H3 FastH3 with an adapter."])

        // An explicit adapter replaces the embedded one, and its recipe replaces FastH3's.
        let adapter = base + ["--h3-adapter", "turbo.safetensors"]
        let adapterOptions = try VideoGenerate.parse(["a", "--model", fastH3, "--h3-adapter", "turbo.safetensors"])
            .makeGenerationOptions(outputURL: URL(fileURLWithPath: "/tmp/a.mp4"))
        #expect(!adapterOptions.usesEmbeddedFastH3Adapter)
        let lifted = try report(adapter + ["--steps", "9", "--h3-acceleration", "balanced", "--image", "a.png", "--end-image", "b.png"])
        #expect(lifted.family == "h3-fast-adapter" && lifted.source == .selector && lifted.violations.isEmpty, "\(lifted)")
        #expect(try report(adapter + ["--reference", "image:a.png"]).violations.count == 1)
    }

    /// The flags that also select a default model get no generated cases; these pin their scope.
    @Test func defaultSelectingFlagsKeepTheirScope() throws {
        func check(_ model: String, _ arguments: [String]) throws -> MereRunFamilyResolutionReport {
            try report(["video", "generate", "a", "--model", model] + arguments)
        }
        // Studio's final-quality default no longer reaches the draft checkpoint; the gate names it.
        #expect(try check("video-ltx23-av-mlx", ["--quality", "final"]).violations
            == ["--quality final is not supported by LTX-2.3 Distilled; it runs draft. Remove --quality or pass draft."])
        #expect(try check("video-ltx25-full-bf16", ["--quality", "final", "--audio", "a.wav"]).violations.isEmpty)
        #expect(try check("video-ltx25-distilled-bf16", ["--audio", "a.wav"]).violations
            == ["--audio is not supported by LTX-2.5 Distilled. It applies to LTX-2.3 Full, LTX-2.3 A2Vid and LTX-2.5 Full."])
        #expect(try check("video-ltx25-distilled-bf16", ["--video-decoder", "diffusion"]).warnings
            == ["--video-decoder diffusion has no effect with LTX-2.5 Distilled; it runs convolutional. "
                + "Remove --video-decoder or pass convolutional."])
        #expect(try check("video-wan22-ti2v-5b-mlx", ["--image", "a.png", "--ltx-preset", "standard"]).violations.isEmpty)
        #expect(try check("video-wan22-ti2v-5b-mlx", ["--image", "a.png", "--ltx-preset", "hq"]).violations
            == ["--ltx-preset hq is not supported by Wan 2.2 TI2V; it runs standard. Remove --ltx-preset or pass standard."])
        #expect(try check("video-wan22-ti2v-5b-mlx", ["--image", "a.png", "--variant", "distilled"]).warnings
            == ["--variant has no effect with Wan 2.2 TI2V. It applies to LTX (merged), LTX-2.3 Distilled, LTX-2.3 Full, "
                + "LTX-2.3 A2Vid, LTX-2.5 Distilled, LTX-2.5 Distilled with the diffusion decoder and LTX-2.5 Full."])
        #expect(try check("video-minimax-h3-fl2va-bf16-mlx", ["--enhance-prompt"]).violations.count == 1)
        #expect(try check("video-ltx23-full-mlx", ["--num-generated-keyframes", "0"]).violations.isEmpty)
        #expect(try check("video-ltx23-full-mlx", ["--num-generated-keyframes", "2"]).violations.count == 1)
        #expect(try check("video-minimax-h3-fl2va-bf16-mlx", ["--end-image", "b.png", "--image", "a.png"]).violations.isEmpty)
    }

    @Test func sessionRejectsTeaCacheOnDistilledCheckpointsBeforeLoad() throws {
        let distilled = try report(["video", "session", "--ltx-teacache"])
        #expect(distilled.family == "ltx23-distilled" && distilled.violations
            == ["--ltx-teacache is not supported by LTX-2.3 Distilled. It applies to LTX-2.5 Full."])
        #expect(try report(["video", "session", "--model", "video-ltx25-full-bf16", "--ltx-teacache"]).warnings.isEmpty)
        #expect(try report(["video", "session", "--model", "video-wan22-ti2v-5b-mlx"]).source == .excluded)
    }

    @Test func retakeWarnsWhenTheDistilledDefaultIgnoresGuidance() throws {
        let base = ["video", "retake", "a", "--source", "s.mp4", "--start-time", "0", "--end-time", "1"]
        let distilled = try report(base + ["--steps", "30", "--negative-prompt", "blur"])
        #expect(distilled.family == "ltx25-distilled" && distilled.violations.isEmpty && distilled.warnings.count == 2)
        #expect(try report(base + ["--model", "video-ltx25-full-bf16", "--steps", "30"]).warnings.isEmpty)
        #expect(try report(base + ["--model", "video-ltx23-full-mlx"]).source == .excluded)
    }

    @Test func worldServeRoutesByBackend() throws {
        let dreamX = try report(["world", "serve"])
        #expect(dreamX.family == "dreamx" && dreamX.model == "video-dreamx-world-5b-ar-mlx")
        let cosmos = try report(["world", "serve", "--backend", "cosmos3"])
        #expect(cosmos.family == "cosmos3" && cosmos.model == "video-cosmos3-edge-mlx")
        // Studio sends the DreamX default id with either backend; Cosmos 3 reads it as its own default.
        let studio = try report([
            "world", "serve", "--backend", "cosmos3", "--model", "video-dreamx-world-5b-ar-mlx",
            "--base-model", "video-wan22-ti2v-5b-mlx"
        ])
        #expect(studio.family == "cosmos3" && studio.violations.isEmpty && studio.warnings
            == ["--base-model has no effect with Cosmos 3 Edge. It applies to DreamX World."])
        #expect(try report(["world", "serve", "--model", "video-cosmos3-edge-mlx"]).source == .unmatched)
        let local = try report(["world", "serve", "--backend", "cosmos3", "--model", "/tmp/cosmos3"])
        #expect(local.family == "cosmos3" && local.source == .selector)
    }

    /// A full checkpoint loads its text-to-video lanes without source `--audio` and then refuses
    /// stage-one previews, and LTX-2.3 refuses IC-LoRA references (`LTXUnifiedAVGenerator.generate`);
    /// the audio-to-video lane drops both. The gate refuses the first before loading and warns on
    /// the second.
    @Test func fullCheckpointsRefuseReferenceControlsOnlyWithoutSourceAudio() throws {
        let full23 = try makeRoot(Self.ltx23A2VidFiles + ["vocoder.safetensors"])
        let a2vid = try makeRoot(Self.ltx23A2VidFiles)
        defer { for root in [full23, a2vid] { try? FileManager.default.removeItem(at: root) } }
        let reference = ["--video-conditioning", "r.mp4", "--lora", "ic.safetensors"]
        let runs: [([String], String)] = [
            (["--model", "video-ltx25-full-bf16"], "LTX-2.5 Full"),
            (["--model-root", full23.path], "LTX-2.3 Full"),
            (["--model-root", a2vid.path], "LTX-2.3 A2Vid")
        ]
        for (model, title) in runs {
            let generate = ["video", "generate", "p"] + model
            let preview = try report(generate + ["--skip-stage-2"] + reference)
            #expect(preview.violations.contains(
                "--skip-stage-2 is not supported by \(title) without --audio. "
                    + "It applies to LTX-2.5 Distilled and LTX-2.5 Distilled with the diffusion decoder."
            ), "\(model): \(preview)")
            #expect(throws: CLICapabilityGate.Rejection.self, "\(model)") {
                try CLICapabilityGate.check(arguments: ["mere.run"] + generate + ["--skip-stage-2"] + reference)
            }
            let withAudio = try report(generate + ["--audio", "a.wav", "--skip-stage-2"] + reference)
            #expect(withAudio.violations.isEmpty, "\(model): \(withAudio)")
            #expect(withAudio.warnings.contains("--skip-stage-2 has no effect with \(title). "
                + "It applies to LTX-2.5 Distilled and LTX-2.5 Distilled with the diffusion decoder."), "\(model)")
            #expect(throws: Never.self, "\(model)") {
                try CLICapabilityGate.check(arguments: ["mere.run"] + generate + ["--audio", "a.wav", "--skip-stage-2"] + reference)
            }
            // A blank --audio is not source audio, as the command reads it.
            #expect(try report(generate + ["--audio", " ", "--skip-stage-2"] + reference).violations.count >= 1, "\(model)")
        }

        for (model, title) in runs.dropFirst() {
            let generate = ["video", "generate", "p"] + model
            let refused = try report(generate + reference)
            #expect(refused.violations == [
                "--video-conditioning is not supported by \(title) without --audio. It applies to LTX-2.5 Distilled, "
                    + "LTX-2.5 Distilled with the diffusion decoder and LTX-2.5 Full."
            ], "\(model): \(refused)")
            let dropped = try report(generate + ["--audio", "a.wav"] + reference)
            #expect(dropped.violations.isEmpty && dropped.warnings.contains { $0.hasPrefix("--video-conditioning has no effect") },
                    "\(model): \(dropped)")
        }
        let full25 = try report(["video", "generate", "p", "--model", "video-ltx25-full-bf16"] + reference)
        #expect(full25.violations.isEmpty && full25.warnings.isEmpty, "\(full25)")
        // The distilled runtimes run both; Wan and MiniMax-H3 accept and ignore both, as before.
        let distilled = try report(["video", "generate", "p", "--model", "video-ltx25-distilled-bf16", "--skip-stage-2"] + reference)
        #expect(distilled.violations.isEmpty && distilled.warnings.isEmpty, "\(distilled)")
        let wan = try report(["video", "generate", "p", "--model", "video-wan22-ti2v-5b-mlx", "--image", "a.png", "--skip-stage-2"] + reference)
        #expect(wan.violations.isEmpty && wan.warnings.count == 3, "\(wan)")
    }

    /// The managed LTX-2.5 Distilled checkpoint has no diffusion decoder and decodes with the
    /// convolutional one whatever is asked; a folder that holds it runs `--video-decoder
    /// diffusion`, in every command that loads the distilled runtime.
    @Test func aDistilledFolderWithTheDiffusionDecoderTakesIt() throws {
        let distilled = LTX25Resources.requiredRelativePaths + [LTX25Resources.textEncoderRelativePath]
        let plain = try makeRoot(distilled)
        let diffusion = try makeRoot(distilled + [LTX25Resources.diffusionVideoVAERelativePath])
        defer { for root in [plain, diffusion] { try? FileManager.default.removeItem(at: root) } }
        let retake = ["video", "retake", "a", "--source", "s.mp4", "--start-time", "0", "--end-time", "1"]
        for command in [["video", "generate", "p"], retake, ["video", "session"]] {
            for spelling in [["--model-root", diffusion.path], ["--model", diffusion.path]] {
                let found = try report(command + spelling + ["--video-decoder", "diffusion"])
                #expect(found.family == "ltx25-distilled-diffusion" && found.violations.isEmpty && found.warnings.isEmpty,
                        "\(command) \(spelling): \(found)")
                #expect(throws: Never.self) { try CLICapabilityGate.check(arguments: ["mere.run"] + command + spelling) }
            }
            let convolutional = try report(command + ["--model-root", plain.path, "--video-decoder", "diffusion"])
            #expect(convolutional.family == "ltx25-distilled" && convolutional.violations.isEmpty
                && convolutional.warnings.count == 1, "\(command): \(convolutional)")
        }
        // Everything else scopes as LTX-2.5 Distilled does.
        let generate = MereRunCapabilityCatalog.videoGenerate
        let generateDistilled = generate.options(forFamily: "ltx25-distilled").map(\.flag)
        #expect(generate.options(forFamily: "ltx25-distilled-diffusion").map(\.flag) == generateDistilled)
        let dfr = try report(["video", "generate", "p", "--model-root", diffusion.path, "--dfr"])
        #expect(dfr.violations == ["--dfr is not supported by LTX-2.5 Distilled with the diffusion decoder. It applies to LTX-2.5 Full."])
        let steps = try report(retake + ["--model-root", diffusion.path, "--steps", "30"])
        #expect(steps.violations.isEmpty && steps.warnings == ["--steps has no effect with LTX-2.5 Distilled with the diffusion decoder. It applies to LTX-2.5 Full."])
        let session = MereRunCapabilityCatalog.videoSession
        #expect(session.options(forFamily: "ltx25-distilled-diffusion").first { $0.flag == "--video-decoder" }?.defaultValue
            == "convolutional")
        #expect(session.options(forFamily: "ltx25-distilled-diffusion").first { $0.flag == "--video-decoder" }?.choices
            == ["convolutional", "diffusion"])
    }

    private func makeRoot(_ files: [String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(path: "video-gate-\(UUID().uuidString)")
        for path in files {
            let file = root.appending(path: path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: file)
        }
        return root
    }

    private static let ltx23A2VidFiles = [
        "split_model.json", "config.json", "connector.safetensors", "transformer-dev.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
        "audio_vae.safetensors", "spatial_upscaler_x2_v1_1.safetensors"
    ]

    private func tokens(_ flag: String, _ value: String) -> [String] {
        let option = MereRunCapabilityCatalog.videoGenerate.options.first { $0.flag == flag }
        return option?.kind == .boolean ? [flag] : [flag, value]
    }

    private func sampleValue(for flag: String) -> String {
        let option = MereRunCapabilityCatalog.videoGenerate.options.first { $0.flag == flag }
        switch (flag, option?.kind) {
        case ("--image-conditioning", _): return "0:a.png"
        case ("--auto-duration", _): return "2"
        case (_, .choice?): return option?.choices.first ?? ""
        case (_, .integer?): return "8"
        case (_, .number?): return "0.5"
        default: return "a.file"
        }
    }
}
