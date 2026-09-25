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
                + "LTX-2.3 Full, LTX-2.3 A2Vid, LTX-2.5 Distilled, LTX-2.5 Full, Wan 2.2 TI2V and MiniMax-H3 FL2VA."])
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
                + "LTX-2.3 A2Vid, LTX-2.5 Distilled and LTX-2.5 Full."])
        #expect(try check("video-minimax-h3-fl2va-bf16-mlx", ["--enhance-prompt"]).violations.count == 1)
        #expect(try check("video-ltx23-full-mlx", ["--num-generated-keyframes", "0"]).violations.isEmpty)
        #expect(try check("video-ltx23-full-mlx", ["--num-generated-keyframes", "2"]).violations.count == 1)
        #expect(try check("video-minimax-h3-fl2va-bf16-mlx", ["--end-image", "b.png", "--image", "a.png"]).violations.isEmpty)
    }

    @Test func sessionRejectsTeaCacheOnDistilledCheckpointsBeforeLoad() throws {
        let distilled = try report(["video", "session", "--ltx-teacache"])
        #expect(distilled.family == "ltx23-distilled" && distilled.violations
            == ["--ltx-teacache is not supported by LTX-2.3 Distilled. It applies to LTX-2.5 Full."])
        #expect(try report(["video", "session", "--model", "video-ltx23-full-mlx", "--ltx-teacache"]).warnings.count == 1)
        #expect(try report(["video", "session", "--model", "video-ltx25-full-bf16", "--ltx-teacache"]).warnings.isEmpty)
        #expect(try report(["video", "session", "--model", "video-wan22-ti2v-5b-mlx"]).source == .excluded)
        #expect(try report(["video", "session", "--model", "video-ltx-av"]).source == .unidentified)
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
