import Foundation
import MereRunContract
import MereRunCore
import Testing

@testable import MereRunCLI

// Command lines that ran before the capability gate and must keep running, one per way the gate
// could misread them: how ArgumentParser spells a command line, a value the command normalizes,
// and a model the command accepts in a form the contract did not list.

private func report(_ commandLine: [String]) throws -> MereRunFamilyResolutionReport {
    try #require(CLICapabilityGate.evaluate(commandLine: commandLine)).report
}

private func expectRuns(
    _ commandLine: [String],
    family: String? = nil,
    warnings: Int? = nil,
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let found = try report(commandLine)
    #expect(found.violations.isEmpty, "\(commandLine): \(found.violations)", sourceLocation: sourceLocation)
    if let family {
        #expect(found.family == family, "\(commandLine): \(found)", sourceLocation: sourceLocation)
    }
    if let warnings {
        #expect(found.warnings.count == warnings, "\(commandLine): \(found.warnings)", sourceLocation: sourceLocation)
    }
    #expect(throws: Never.self, "\(commandLine)", sourceLocation: sourceLocation) {
        try CLICapabilityGate.check(arguments: ["mere.run"] + commandLine)
    }
}

// MARK: - Spellings ArgumentParser accepts

/// ArgumentParser keeps the last occurrence of a single-value option; the gate reads that one.
@Test func aRepeatedSingleValueOptionReadsAsItsLastValue() throws {
    try expectRuns(["audio", "enhance", "a.wav", "--input-rate", "8000", "--input-rate", "16000"], family: "ap-bwe")
    #expect(try report(["audio", "enhance", "a.wav", "--input-rate", "16000", "--input-rate", "8000"]).violations.count == 1)
    try expectRuns(["music", "generate", "song", "--model", "music-yue2", "--vocal-language", "fr", "--vocal-language", "en"],
                   family: "yue2")
    try expectRuns(["text", "chat", "-p", "hi", "--model", "vision-chat-q38-27b-4bit", "--reasoning-effort", "5",
                    "--reasoning-effort", "0.5"], family: "q38", warnings: 0)
    try expectRuns(["text", "chat", "-p", "hi", "--model", "text-chat-q36-nano", "--kv-bits", "3", "--kv-bits", "4"])
}

/// A single-dash group is its short options, each taking the next value in order.
@Test func aGroupOfShortOptionsReadsAsEachOfThem() throws {
    try expectRuns(["music", "generate", "a song", "-qm", "music-yue2", "--semantic-top-p", "0.9"], family: "yue2", warnings: 0)
    try expectRuns(["music", "generate", "a song", "-mq", "music-yue2", "--semantic-top-p", "0.9"], family: "yue2", warnings: 0)
    try expectRuns(["music", "generate", "a song", "-m=music-yue2", "--semantic-top-p", "0.9"], family: "yue2", warnings: 0)
}

/// Choices the CLI trims, lowercases, or reads as numbers.
@Test func choicesTheCLINormalizesReadAsTheCLIReadsThem() throws {
    for scheme in ["UNIFORM", " uniform", "Uniform "] {
        try expectRuns(["text", "chat", "-p", "hi", "--model", "text-chat-q36-nano", "--kv-bits", "8", "--kv-quant-scheme", scheme])
    }
    for rate in ["016000", "+16000", "16000"] {
        try expectRuns(["audio", "enhance", "a.wav", "--input-rate", rate], family: "ap-bwe")
    }
}

// MARK: - Values the commands read as not passed

/// Text chat trims a media path or adapter to nothing and reads it as not passed.
@Test func aBlankChatAttachmentReadsAsOmitted() throws {
    for flag in ["--image", "--audio", "--video", "--lora"] {
        for blank in ["", "  "] {
            try expectRuns(["text", "chat", "-p", "hi", "--model", "text-chat-gemma4-nano", flag, blank], family: "gemma4")
            try expectRuns(["text", "chat", "-p", "hi", "--model", "text-chat-diffusiongemma-26b-optiq-4bit", flag, blank])
        }
    }
}

/// Video generate reads blank `--audio` as no source audio: it neither picks the full checkpoint
/// nor refuses a draft run.
@Test func blankSourceAudioDoesNotRouteVideo() throws {
    let blank = try report(["video", "generate", "a wave", "--audio", ""])
    #expect(blank.family == "ltx23-distilled" && blank.violations.isEmpty, "\(blank)")
    try expectRuns(["video", "generate", "a wave", "--audio", "", "--quality", "draft"], family: "ltx23-distilled")
    #expect(try report(["video", "generate", "a wave", "--audio", "a.wav"]).family == "ltx23-full")
}

/// `--stems` splits on commas and drops blanks, so a list with no stem left does nothing.
@Test func anEmptyStemListReadsAsOmitted() throws {
    for stems in [",", " , ", ""] {
        try expectRuns(["music", "generate", "x", "--model", "music-acestep", "--stems", stems], family: "ace-step-turbo", warnings: 1)
    }
    #expect(try report(["music", "generate", "x", "--model", "music-acestep", "--stems", "vocals"]).violations.count == 1)
}

/// `--flow-edit` runs text-to-music whatever `--task-type` says, so the task only warns.
@Test func flowEditReplacesTheTaskOnEveryACEStepFamily() throws {
    let flowEdit = ["--flow-edit", "--source-audio", "a.wav", "--source-caption", "y"]
    for model in ["music-acestep", "music-acestep-xl-sft", "music-acestep-xl-base"] {
        let found = try report(["music", "generate", "x", "--model", model, "--task-type", "extract"] + flowEdit)
        #expect(found.violations.isEmpty, "\(model): \(found)")
        #expect(found.warnings == ["--task-type extract has no effect with --flow-edit; it runs text2music."], "\(model)")
        try expectRuns(["music", "generate", "x", "--model", model, "--task-type", "text2music"] + flowEdit, warnings: 0)
    }
    #expect(try report(["music", "generate", "x", "--model", "music-acestep", "--task-type", "extract"]).violations.count == 1)
}

// MARK: - Default routing

/// Any positive keyframe count routes to LTX-2.5 Distilled, as the command's `> 0` check does.
@Test func anyGeneratedKeyframeCountPicksLTX25Distilled() throws {
    for count in ["1", "16", "17", "20", "64"] {
        try expectRuns(["video", "generate", "p", "--num-generated-keyframes", count], family: "ltx25-distilled")
    }
    #expect(try report(["video", "generate", "p", "--num-generated-keyframes", "0"]).family == "ltx23-distilled")
}

// MARK: - Models the commands accept

/// A model the live listener replaces, and every listing flag, run as before.
@Test func musicListingFlagsAnswerBeforeAnyModel() throws {
    try expectRuns(["music", "transcribe", "--list-instruments", "--model", "music-acestep-lm-4b"])
    try expectRuns(["music", "realtime", "--list-midi-inputs", "--model", "music-acestep"])
}

/// The ACE-Step commands load `--checkpoints-root` before they ever resolve `--model`, so a
/// language model id there is never loaded and does not refuse the run.
@Test func aCheckpointsRootStandsInForALanguageModelId() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).path
    for lm in ["music-acestep-lm-1.7b", "music-acestep-lm-4b"] {
        try expectRuns(["music", "analyze", "a.wav", "--model", lm, "--checkpoints-root", root], family: "ace-step")
        try expectRuns(["music", "train-adapter", "--model", lm, "--checkpoints-root", root, "--dataset", "d", "--output", "o"],
                       family: "ace-step")
        try expectRuns(["music", "serve", "--model", lm, "--checkpoints-root", root], family: "ace-step")
        for command in [["music", "analyze", "a.wav"], ["music", "serve"], ["music", "generate", "x"]] {
            let refused = try report(command + ["--model", lm])
            #expect(refused.source == .excluded && refused.violations.count == 1, "\(command) \(lm)")
        }
    }
}

/// With nothing installed, a managed ACE-Step id under the default decoder runs its own layout.
@Test func aManagedACEStepIdKeepsItsFamilyUnderTheDefaultDecoder() throws {
    let base = try report(["music", "generate", "x", "--model", "music-acestep-xl-base", "--decoder-subdirectory", "acestep-v15-turbo"])
    #expect(base.family == "ace-step-base" || base.source == .identified, "\(base)")
    #expect(base.source != .unidentified, "\(base)")
    let named = try report(["music", "generate", "x", "--model", "music-acestep-xl-base", "--decoder-subdirectory", "acestep-v15-xl-sft"])
    #expect(named.violations.isEmpty)
}

/// Retake asks for the diffusion decoder by default, and the session decodes LTX-2.5 Full with
/// it; LTX-2.5 Distilled installs only the convolutional decoder. Each family shows what it runs.
@Test func videoDecoderDefaultsFollowEachCommand() throws {
    func decoder(_ capability: MereRunCommandCapability, _ family: String) -> String? {
        capability.options(forFamily: family).first { $0.flag == "--video-decoder" }?.defaultValue
    }
    for capability in [MereRunCapabilityCatalog.videoRetake, MereRunCapabilityCatalog.videoSession] {
        #expect(decoder(capability, "ltx25-full") == "diffusion", "\(capability.id)")
        #expect(decoder(capability, "ltx25-distilled") == "convolutional", "\(capability.id)")
    }
    try expectRuns(["video", "session", "--model", "video-ltx25-full-bf16", "--video-decoder", "diffusion"], warnings: 0)
}

/// Every ACE-Step checkpoint analyzes and trains adapters; LTX-2.5 Full dubs.
@Test func pickersOfferEveryCheckpointTheCommandRuns() throws {
    for model in ["music-acestep", "music-acestep-xl-turbo", "music-acestep-xl-turbo-lm4b", "music-acestep-xl-sft", "music-acestep-xl-base"] {
        #expect(try report(["music", "analyze", "a.wav", "--model", model]).family == "ace-step", "\(model)")
        #expect(try report(["music", "train-adapter", "--model", model, "--dataset", "d", "--output", "o"]).family == "ace-step")
    }
    #expect(try report(["video", "dub-it", "a.mp4", "--model", "video-ltx25-full-bf16"]).family == "ltx25-full")
}

// MARK: - Families split where the runtime splits

/// The legacy 4-bit FL2VA checkpoint refuses Turbo adapters before loading, as the run did after.
@Test func theLegacyQ4FL2VACheckpointRefusesAdapters() throws {
    let q4 = ["video", "generate", "p", "--model", "video-minimax-h3-fl2va-mlx"]
    try expectRuns(q4 + ["--image", "a.png"], family: "h3-fl2va-q4", warnings: 0)
    #expect(try report(q4 + ["--h3-adapter", "turbo.safetensors"]).violations
        == ["--h3-adapter is not supported by MiniMax-H3 FL2VA 4-bit. It applies to MiniMax-H3 FL2VA, "
            + "MiniMax-H3 FastH3, MiniMax-H3 FastH3 with an adapter and MiniMax-H3 Ref2VA."])
    try expectRuns(["video", "generate", "p", "--model", "video-minimax-h3-fl2va-8bit-mlx", "--h3-adapter", "turbo.safetensors"],
                   family: "h3-fl2va")
}

/// LFM2.5 loads a text adapter only on the 8-bit A1B runtime; the others fail after loading, so
/// the gate refuses them first. A context below 512 runs on every LFM2.5 checkpoint.
@Test func lfmTextAdaptersRunOnlyOnTheEightBitA1B() throws {
    try expectRuns(["text", "chat", "-p", "hi", "--model", "text-chat-lfm25-a1b-8bit", "--lora", "a.safetensors"],
                   family: "lfm2-a1b", warnings: 0)
    for model in ["text-chat-lfm25-a1b-bf16", "text-chat-lfm25-1.2b-bf16", "vision-chat-lfm25-3b-8bit"] {
        #expect(try report(["text", "chat", "-p", "hi", "--model", model, "--lora", "a.safetensors"]).violations.count == 1, "\(model)")
        try expectRuns(["text", "chat", "-p", "hi", "--model", model, "--context-size", "256"], warnings: 0)
    }
    #expect(try report(["text", "chat", "-p", "hi", "--model", "text-chat-lfm25-1.2b-bf16", "--context-size", "65536"]).warnings.count == 1)
}
