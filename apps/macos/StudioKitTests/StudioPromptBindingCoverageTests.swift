@testable import StudioKit
import MereRunContract
import XCTest

/// Every essential and standard option a prompt task's command declares is editable from that
/// task's inspector or composer: it has a binding in `StudioContractBindings`, a composite editor
/// in `StudioContractOverrides`, or a reason below why the prompt task leaves it to the CLI.
final class StudioPromptBindingCoverageTests: XCTestCase {
    /// Declared essential or standard flags no prompt task binds, because Studio sets them itself.
    /// Keep each reason to one line.
    private static let studioOwned: [String: String] = [
        "--prompt": "The composer's prompt field writes it.",
        "--query": "The composer's prompt field writes it, one query per line.",
        "--output": "Studio names the output in its Library folders (StudioOutputLocation).",
        "--output-dir": "Studio names the output in its Library folders (StudioOutputLocation).",
        "--json-output": "Studio always writes the result document it renders (StudioVisionResultPaths).",
        "--mask-output-dir": "Studio always writes the mask sidecars it renders (StudioVisionResultPaths).",
        "--stream": "Conversation turns always stream so the transcript renders tokens as they arrive.",
    ]

    /// Declared essential or standard flags one prompt task deliberately does not bind.
    private static let unbound: [StudioMode: [String: String]] = [
        .chat: ["--markdown": "It styles terminal output; Studio renders the streamed reply itself."],
        .listen: ["--timestamps": "--no-timestamps is bound to the same draft field, and timestamps are on by default."],
    ]

    private static func reason(for flag: String, mode: StudioMode) -> String? {
        unbound[mode]?[flag] ?? studioOwned[flag]
    }

    /// The capabilities each prompt task runs. Read Image runs Inspect and Caption.
    private static func capabilities(for mode: StudioMode) -> [MereRunCommandCapability] {
        let actions: [StudioReadImageAction] = mode == .readImage ? [.inspect, .caption] : [.inspect]
        return actions.compactMap { StudioContractSchema.capability(for: mode, readImageAction: $0) }
    }

    private static func isBound(_ flag: String, mode: StudioMode) -> Bool {
        StudioContractBindings.bindings(for: mode)[flag] != nil
            || StudioContractOverrides.override(forFlag: flag, mode: mode) != nil
    }

    func testEveryEssentialAndStandardOptionHasABindingOrAReason() {
        for mode in StudioMode.allCases {
            let capabilities = Self.capabilities(for: mode)
            XCTAssertFalse(capabilities.isEmpty, "\(mode) runs no contract capability")
            for capability in capabilities {
                for option in capability.options where (option.tier ?? .standard) != .expert {
                    guard !Self.isBound(option.flag, mode: mode) else { continue }
                    XCTAssertNotNil(
                        Self.reason(for: option.flag, mode: mode),
                        "\(mode) (\(capability.id)) declares \(option.flag) as \(option.tier ?? .standard) but binds no draft "
                            + "field to it. Bind it in StudioContractBindings or give the reason in `unbound`."
                    )
                }
            }
        }
    }

    func testEveryReasonNamesADeclaredUnboundOption() {
        func isUnboundDeclared(_ flag: String, mode: StudioMode) -> Bool {
            let option = Self.capabilities(for: mode).flatMap(\.options).first { $0.flag == flag }
            return option.map { ($0.tier ?? .standard) != .expert } == true && !Self.isBound(flag, mode: mode)
        }
        for (mode, reasons) in Self.unbound {
            for flag in reasons.keys {
                XCTAssertTrue(isUnboundDeclared(flag, mode: mode), "\(mode) lists \(flag), which it binds or does not declare as essential or standard")
            }
        }
        for flag in Self.studioOwned.keys {
            XCTAssertTrue(
                StudioMode.allCases.contains { isUnboundDeclared(flag, mode: $0) },
                "No prompt task declares \(flag) unbound, so remove it from `studioOwned`"
            )
        }
    }

    // MARK: - Bound options reach the command

    func testMusicComposeSendsTempoKeyMeterVocalLanguageAndGuidance() throws {
        let argv = try arguments(.music, model: "music-acestep-xl-sft", [
            "--bpm": .integer(120),
            "--keyscale": .text("C major"),
            "--timesignature": .text("3"),
            "--vocal-language": .text("ja"),
            "--guidance-scale": .number(6.5),
        ])
        assertPair(argv, "--bpm", "120")
        assertPair(argv, "--keyscale", "C major")
        assertPair(argv, "--timesignature", "3")
        assertPair(argv, "--vocal-language", "ja")
        assertPair(argv, "--guidance-scale", "6.5")
    }

    func testVoiceSpeakSendsLanguageAndTemperature() throws {
        let argv = try arguments(.speak, ["--language": .text("en"), "--temperature": .number(0.9)])
        assertPair(argv, "--language", "en")
        assertPair(argv, "--temperature", "0.9")
    }

    func testTranscribeSendsTaskProviderBeamSizeAndKeywords() throws {
        assertPair(try arguments(.listen, ["--task": .text("translate")]), "--task", "translate")
        assertPair(
            try arguments(.listen, model: "speech-asr-parakeet", ["--provider": .text("coreml")]),
            "--provider", "coreml"
        )
        let whistle = try arguments(.listen, model: "speech-asr-whistle", [
            "--beam-size": .integer(3),
            "--keyword": .text("mere.run, Studio"),
        ])
        assertPair(whistle, "--beam-size", "3")
        assertPair(whistle, "--keyword", "mere.run")
        assertPair(whistle, "--keyword", "Studio")
    }

    func testVideoGenerateSendsLoRAsAndPromptEnhancement() throws {
        let argv = try arguments(.video, model: "video-ltx25-full-bf16", [
            "--enhance-prompt": .flag(true),
            "--lora": .text("/tmp/style.safetensors=0.8, /tmp/motion.safetensors"),
        ])
        XCTAssertTrue(argv.contains("--enhance-prompt"), argv.joined(separator: " "))
        assertPair(argv, "--lora", "/tmp/style.safetensors=0.8")
        assertPair(argv, "--lora", "/tmp/motion.safetensors")
    }

    func testSoundGenerateSendsNegativePromptAndCFG() throws {
        let argv = try arguments(.sfx, model: "sfx-mmaudio-large-44k-v2", [
            "--negative-prompt": .text("wind noise"),
            "--cfg": .number(6),
        ])
        assertPair(argv, "--negative-prompt", "wind noise")
        assertPair(argv, "--cfg", "6")
        XCTAssertFalse(try arguments(.sfx, [:]).contains("--cfg"), "an untouched CFG follows the model")
    }

    func testSegmentAndTrackSendBoxesAndMultipleMasks() throws {
        let segment = try arguments(.segment, ["--show-boxes": .flag(true), "--multimask": .flag(true)])
        XCTAssertTrue(segment.contains("--show-boxes"), segment.joined(separator: " "))
        XCTAssertTrue(segment.contains("--multimask"), segment.joined(separator: " "))
        let track = try arguments(.track, ["--show-boxes": .flag(true)])
        XCTAssertTrue(track.contains("--show-boxes"), track.joined(separator: " "))
        let untouched = try arguments(.segment, [:])
        XCTAssertFalse(untouched.contains("--show-boxes") || untouched.contains("--multimask"))
    }

    func testTranscribeSendsMaxTokens() throws {
        assertPair(try arguments(.listen, model: "speech-asr-qwen3", ["--max-tokens": .integer(900)]), "--max-tokens", "900")
    }

    func testCodeSendsShowThinkingOnlyWhenOn() throws {
        XCTAssertTrue(try arguments(.code, ["--thinking": .flag(true)]).contains("--thinking"))
        XCTAssertFalse(try arguments(.code, [:]).contains("--thinking"))
    }

    func testReadImageSendsChosenSamplingAndKeepsEachActionsDefaults() throws {
        let inspect = try arguments(.readImage, ["--max-tokens": .integer(512), "--temperature": .number(0.3), "--top-p": .number(0.8)])
        assertPair(inspect, "--max-tokens", "512")
        assertPair(inspect, "--temperature", "0.3")
        assertPair(inspect, "--top-p", "0.8")

        let caption = try arguments(.readImage, [
            "--focus": .text("lighting"),
            "--trigger-token": .text("mrstyle"),
        ]) { $0.readImageAction = .caption }
        assertPair(caption, "--focus", "lighting")
        assertPair(caption, "--trigger-token", "mrstyle")
        let untouched = try arguments(.readImage, [:]) { $0.readImageAction = .caption }
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionCaption))
        assertPair(untouched, "--max-tokens", String(template.defaultDraft().maxTokens))
        XCTAssertFalse(untouched.contains("--temperature"), "an untouched temperature follows Caption's default")
    }

    /// The argv a prompt task launches after its inspector writes `values` through the bindings.
    private func arguments(
        _ mode: StudioMode,
        model: String? = nil,
        _ values: [String: StudioContractValue],
        prepare: (inout StudioDraft) -> Void = { _ in }
    ) throws -> [String] {
        var draft = StudioDraft.baseline(for: mode)
        draft.prompt = "a prompt"
        draft.inputPath = "/tmp/input"
        if let model { draft.model = model }
        prepare(&draft)
        let bindings = StudioContractBindings.bindings(for: mode)
        for (flag, value) in values {
            try XCTUnwrap(bindings[flag], "\(mode) has no binding for \(flag)").write(&draft, value)
        }
        let request = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft, validating: false, source: .contract)
        return request.template.arguments(from: request.draft, source: .contract)
    }

    private func assertPair(_ argv: [String], _ flag: String, _ value: String, line: UInt = #line) {
        let found = zip(argv, argv.dropFirst()).contains { $0 == flag && $1 == value } || argv.contains("\(flag)=\(value)")
        XCTAssertTrue(found, "expected \(flag) \(value) in: \(argv.joined(separator: " "))", line: line)
    }
}
