import AudioCore
import AudioSTT
import AudioTTS
import Foundation
import MereRunContract
import MereRunCore
import Testing

@testable import MereRunCLI

// The generated gate cases skip routing flags (`--backend`, `--task`, `--mode`, `--model`), so
// the routing each speech command takes from them is pinned here, together with its agreement
// with the routers the commands run.

private func report(_ commandLine: String...) throws -> MereRunFamilyResolutionReport {
    try #require(CLICapabilityGate.evaluate(commandLine: commandLine)).report
}

private func temporaryFolder() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

// MARK: - speech transcribe

@Test func transcribeDefaultsFollowTheTaskAndBackend() throws {
    let blank = try report("speech", "transcribe", "a.wav")
    #expect(blank.family == "parakeet" && blank.model == "speech-asr-parakeet" && blank.source == .defaultModel)
    for flags in [["--backend", "qwen"], ["--task", "translate"], ["--task", "translate", "--backend", "auto"]] {
        let qwen = try #require(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe", "a.wav"] + flags)).report
        #expect(qwen.family == "qwen3-asr" && qwen.model == "speech-asr-qwen3", "\(flags)")
        #expect(qwen.violations.isEmpty && qwen.warnings.isEmpty, "\(flags)")
    }
    let named = try report("speech", "transcribe", "a.wav", "-m", "speech-asr-qwen3")
    #expect(named.family == "qwen3-asr" && named.source == .model && named.warnings.isEmpty)
}

/// Translation outranks an explicit `--backend parakeet`: the CLI runs Qwen3-ASR, as it always
/// has for a file, and now says the backend flag had no effect.
@Test func translationOutranksAnExplicitParakeetBackendWithAWarning() throws {
    let translated = try report("speech", "transcribe", "a.wav", "--backend", "parakeet", "--task", "translate")
    #expect(translated.family == "qwen3-asr" && translated.violations.isEmpty)
    #expect(translated.warnings == ["--backend parakeet has no effect with Qwen3-ASR; use auto or qwen."])
}

/// Translation always targets English, so a language hint still routes but changes nothing else.
@Test func translationSaysALanguageHintHasNoEffect() throws {
    let translated = try report("speech", "transcribe", "a.wav", "--task", "translate", "--language", "German")
    #expect(translated.family == "qwen3-asr" && translated.violations.isEmpty)
    #expect(translated.warnings == ["--language German has no effect with --task translate."])
    let transcribed = try report("speech", "transcribe", "a.wav", "--backend", "qwen", "--language", "German")
    #expect(transcribed.family == "qwen3-asr" && transcribed.warnings.isEmpty)
}

/// A named managed model the other flags overrule is swapped for the chosen backend's default,
/// as `SpeechTranscriptionResolver` does, with a warning instead of a silent swap.
@Test func aManagedModelTheFlagsOverruleIsReplacedWithAWarning() throws {
    let cases: [([String], family: String, model: String, warning: String)] = [
        (["--model", "speech-asr-parakeet", "--task", "translate"], "qwen3-asr", "speech-asr-qwen3",
         "--model speech-asr-parakeet has no effect: the other options select Qwen3-ASR."),
        (["--model", "speech-asr-parakeet", "--backend", "qwen"], "qwen3-asr", "speech-asr-qwen3",
         "--model speech-asr-parakeet has no effect: the other options select Qwen3-ASR."),
        (["--model", "speech-asr-qwen3", "--backend", "parakeet"], "parakeet", "speech-asr-parakeet",
         "--model speech-asr-qwen3 has no effect: the other options select Parakeet."),
        (["-m", "mlx-community/parakeet-tdt-0.6b-v3", "--task", "translate"], "qwen3-asr", "speech-asr-qwen3",
         "--model mlx-community/parakeet-tdt-0.6b-v3 has no effect: the other options select Qwen3-ASR.")
    ]
    for (flags, family, model, warning) in cases {
        let replaced = try #require(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe", "a.wav"] + flags)).report
        #expect(replaced.family == family && replaced.model == model && replaced.source == .defaultModel, "\(flags)")
        #expect(replaced.violations.isEmpty && replaced.warnings == [warning], "\(flags): \(replaced.warnings)")
    }
    // The model Qwen3-ASR runs anyway draws no model warning, only the overruled backend does.
    let kept = try report("speech", "transcribe", "a.wav", "--model", "speech-asr-qwen3", "--backend", "parakeet",
                          "--task", "translate")
    #expect(kept.family == "qwen3-asr" && kept.model == "speech-asr-qwen3")
    #expect(kept.warnings == ["--backend parakeet has no effect with Qwen3-ASR; use auto or qwen."])
}

@Test func transcribeRejectsCoreMLWhereverQwenRuns() throws {
    let translated = try report("speech", "transcribe", "a.wav", "--backend", "parakeet", "--task", "translate",
                                "--provider", "coreml", "--coreml-encoder", "/tmp/parakeet-coreml")
    #expect(translated.violations == [
        "--provider is not supported by Qwen3-ASR. It applies to Parakeet.",
        "--coreml-encoder is not supported by Qwen3-ASR. It applies to Parakeet."
    ])
    #expect(throws: CLICapabilityGate.Rejection.self) {
        try CLICapabilityGate.check(arguments: ["mere.run", "speech", "transcribe", "a.wav", "--backend", "qwen",
                                                "--provider", "coreml", "--coreml-encoder", "/tmp/parakeet-coreml"])
    }
    let parakeet = try report("speech", "transcribe", "a.wav", "--backend", "parakeet", "--provider", "coreml",
                              "--coreml-encoder", "/tmp/parakeet-coreml")
    #expect(parakeet.family == "parakeet" && parakeet.violations.isEmpty && parakeet.warnings.isEmpty)
}

/// A language Parakeet's router does not recognize runs Qwen3-ASR ahead of an explicit backend or
/// model, as the CLI always has; the gate asks that router, so it says so and warns.
@Test func anUnrecognizedLanguageRunsQwenAheadOfAnExplicitChoice() throws {
    let backend = try report("speech", "transcribe", "a.wav", "--backend", "parakeet", "--language", "zz")
    #expect(backend.family == "qwen3-asr" && backend.model == "speech-asr-qwen3" && backend.violations.isEmpty)
    #expect(backend.warnings == ["--backend parakeet has no effect with Qwen3-ASR; use auto or qwen."])
    let model = try report("speech", "transcribe", "a.wav", "--model", "speech-asr-parakeet", "--language", "auto")
    #expect(model.family == "qwen3-asr" && model.model == "speech-asr-qwen3" && model.source == .defaultModel)
    #expect(model.warnings == ["--model speech-asr-parakeet has no effect: the other options select Qwen3-ASR."])
    let known = try report("speech", "transcribe", "a.wav", "--backend", "parakeet", "--language", "English")
    #expect(known.family == "parakeet" && known.warnings.isEmpty)
}

/// The gate names the backend and model the command's router runs, for files and streams alike
/// (both route through `SpeechTranscriptionResolver.route`), on every model spelling, backend,
/// task, and language, with no exceptions.
@Test func transcribeRoutingAgreesWithTheSpeechResolver() throws {
    let managed = try #require(MereRunCapabilityCatalog.speechTranscribe.routing).families.flatMap(\.models)
    let models: [String?] = [nil] + managed + ["mlx-community/parakeet-tdt-0.6b-v3", "fixture/custom-asr"]
    let languages: [String?] = [nil, "en", "English", "zh-CN", "<|fr|>", "auto", "zz", "klingon-ish"]
    for model in models {
        for backend in ASRBackend.allCases {
            for task in [ASRTask.transcribe, .translate] {
                for language in languages {
                    var arguments = ["a.wav", "--backend", backend.rawValue, "--task", task.rawValue]
                    if let model { arguments += ["--model", model] }
                    if let language { arguments += ["--language", language] }
                    let context = arguments.joined(separator: " ")
                    let route = try SpeechTranscriptionResolver.route(
                        task: task, language: language, preferredBackend: backend, modelOverride: model
                    )
                    let parakeet = route.decision.backend == .parakeet
                    let runs = route.modelOverride ?? (parakeet ? ParakeetResources.defaultModelId : Qwen3ASRResources.defaultModelId)
                    let report = try #require(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe"] + arguments)).report
                    #expect(report.family == (parakeet ? "parakeet" : "qwen3-asr"), "\(context)")
                    #expect(report.model == (ManagedModelCatalog.spec(for: runs)?.id ?? runs), "\(context)")
                    #expect(report.violations.isEmpty, "\(context): \(report.violations)")
                }
            }
        }
    }
}

// MARK: - speech diarize

@Test func diarizeRefusesAStreamingBufferForSortformerIncludingTheDefault() throws {
    let refusal = "--latency is not supported by Sortformer. It applies to Nemotron 3 Diarization."
    for model in [[], ["--model", "speech-diarization-sortformer"]] {
        let sortformer = try #require(
            CLICapabilityGate.evaluate(commandLine: ["speech", "diarize", "a.wav", "--latency", "0.64"] + model)
        ).report
        #expect(sortformer.family == "sortformer" && sortformer.violations == [refusal], "\(model)")
    }
    let nemotron = try report("speech", "diarize", "a.wav", "--model", "speech-diarization-nemotron3", "--latency", "0.64")
    #expect(nemotron.family == "nemotron3" && nemotron.violations.isEmpty)
}

/// A local folder is identified by the detector the command itself uses.
@Test func diarizeIdentifiesALocalFolderByItsNeMoArchive() throws {
    let root = try temporaryFolder()
    defer { try? FileManager.default.removeItem(at: root) }
    let sortformer = try report("speech", "diarize", "a.wav", "--model", root.path, "--latency", "0.32")
    #expect(sortformer.family == "sortformer" && sortformer.source == .identified && sortformer.violations.count == 1)
    #expect(!Nemotron3DiarizationResources.isNemotron3(model: root.path, root: root))

    try Data().write(to: root.appendingPathComponent(Nemotron3DiarizationResources.archivePin.filename))
    let nemotron = try report("speech", "diarize", "a.wav", "--model", root.path, "--latency", "0.32")
    #expect(nemotron.family == "nemotron3" && nemotron.source == .identified && nemotron.violations.isEmpty)
    #expect(Nemotron3DiarizationResources.isNemotron3(model: root.path, root: root))

    let file = root.appendingPathComponent(Nemotron3DiarizationResources.archivePin.filename)
    #expect(try report("speech", "diarize", "a.wav", "--model", file.path).source == .unidentified)
}

// MARK: - speech synthesize

@Test func synthesizeScopesOptionsByModeAndCheckpoint() throws {
    let style = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--voice", "calm", "--profile", "narrator")
    #expect(style.family == "style" && style.model == "speech-tts-qwen3-nano" && style.violations.isEmpty)
    #expect(style.warnings == ["--profile has no effect with Qwen3-TTS style. It applies to Qwen3-TTS clone."])

    let clone = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--mode", "clone", "-v", "calm",
                           "--ref-audio", "me.wav", "--model", "speech-tts-qwen3-customvoice")
    #expect(clone.family == "clone" && clone.model == "speech-tts-qwen3-customvoice")
    #expect(clone.warnings == [
        "--voice has no effect with Qwen3-TTS clone. It applies to Qwen3-TTS style and Qwen3-TTS CustomVoice."
    ])

    let speaker = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--model", "speech-tts-qwen3-customvoice",
                             "--speaker", "Ryan", "--voice", "whispering")
    #expect(speaker.family == "custom-voice" && speaker.source == .selector)
    #expect(speaker.violations.isEmpty && speaker.warnings.isEmpty)
    let alias = try report("speech", "synthesize", "Hi", "-o", "a.wav", "-m", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                           "--speaker", "vivian")
    #expect(alias.family == "custom-voice" && alias.model == "speech-tts-qwen3-customvoice" && alias.warnings.isEmpty)
}

/// Only CustomVoice has named speakers, and only in style mode: anywhere else `--speaker` has no
/// effect and the run goes on.
@Test func synthesizeWarnsAboutASpeakerTheCheckpointOrModeIgnores() throws {
    let nano = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--speaker", "ryan")
    #expect(nano.family == "style" && nano.violations.isEmpty)
    #expect(nano.warnings == ["--speaker has no effect with Qwen3-TTS style. It applies to Qwen3-TTS CustomVoice."])
    let clone = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--mode", "clone", "--ref-audio", "me.wav",
                           "--model", "speech-tts-qwen3-customvoice", "--speaker", "ryan")
    #expect(clone.family == "clone" && clone.violations.isEmpty)
    #expect(clone.warnings == ["--speaker has no effect with Qwen3-TTS clone. It applies to Qwen3-TTS CustomVoice."])
    #expect(throws: Never.self) {
        try CLICapabilityGate.check(arguments: ["mere.run", "speech", "synthesize", "Hi", "-o", "a.wav", "--speaker", "ryan"])
    }
}

/// A local folder is identified by its config, as the command reads it: speakers in
/// `talker_config.spk_id` make it CustomVoice in style mode; `--mode clone` runs any checkpoint.
@Test func synthesizeIdentifiesALocalCheckpointByItsSpeakers() throws {
    let root = try temporaryFolder()
    defer { try? FileManager.default.removeItem(at: root) }
    let config = root.appendingPathComponent("config.json")
    try Data(#"{"tts_model_type": "base", "talker_config": {"spk_id": {}}}"#.utf8).write(to: config)
    let base = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--model", root.path, "--speaker", "ryan")
    #expect(base.family == "style" && base.source == .identified && base.warnings.count == 1)
    #expect(try Qwen3TTSResources(rootURL: root).speaker(named: "ryan") == nil)

    try Data(#"{"tts_model_type": "custom_voice", "talker_config": {"spk_id": {"my_voice": 3100}}}"#.utf8).write(to: config)
    let custom = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--model", root.path, "--speaker", "My_Voice")
    #expect(custom.family == "custom-voice" && custom.source == .identified && custom.warnings.isEmpty)
    #expect(try Qwen3TTSResources(rootURL: root).speaker(named: "My_Voice") == "my_voice")
    let clone = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--mode", "clone", "--model", root.path)
    #expect(clone.family == "clone" && clone.source == .identified)

    let missing = root.appendingPathComponent("missing").path
    #expect(try report("speech", "synthesize", "Hi", "-o", "a.wav", "--model", missing).source == .unidentified)
}

/// Every managed model the contract lists is one the command's model selection takes; style mode
/// splits them by checkpoint, and clone runs both.
@Test func synthesizeModelsAgreeWithTheCommandsModelSelection() throws {
    let routing = try #require(MereRunCapabilityCatalog.speechSynthesize.routing)
    for family in routing.families {
        for model in family.models {
            #expect(try SpeechSynthesisModelSelection.resolve(model).modelID == model, "\(family.id) \(model)")
        }
    }
    #expect(routing.family(id: "style")?.models == [Qwen3TTSResources.defaultModelId])
    #expect(routing.family(id: "custom-voice")?.models == [Qwen3TTSResources.customVoiceModelId])
    #expect(Set(routing.family(id: "clone")?.models ?? []) == Qwen3TTSResources.supportedModelIds)
}

/// The speakers Studio offers are the published CustomVoice checkpoint's.
@Test func synthesizeSpeakerChoicesAreTheCustomVoiceCheckpoints() throws {
    let speaker = try #require(MereRunCapabilityCatalog.speechSynthesize.options.first { $0.flag == "--speaker" })
    #expect(speaker.choices == Qwen3TTSResources.customVoiceSpeakers)
    #expect(speaker.families == ["custom-voice"] && Set(speaker.ignoredBy) == ["style", "clone"])
}

/// A named speaker already has a voice: the default description is not sent with one, a
/// description the user wrote is, and without a speaker nothing changes.
@Test func synthesizeSendsTheVoiceDescriptionWithASpeakerOnlyWhenItWasWritten() throws {
    let output = URL(fileURLWithPath: "/tmp/a.wav")
    let plan = { (arguments: [String]) in try SpeechSynthesize.parse(["Hi", "-o", "a.wav"] + arguments).synthesisPlan(outputURL: output) }
    let named = try plan(["--speaker", "Ryan"]).request
    #expect(named.speaker == "Ryan" && named.voiceDescription.isEmpty)
    let instructed = try plan(["--speaker", "ryan", "--voice", "Very angry"]).request
    #expect(instructed.speaker == "ryan" && instructed.voiceDescription == "Very angry")
    let described = try plan([]).request
    #expect(described.speaker == nil && described.voiceDescription == TTSRequest.defaultVoiceDescription)
}

// MARK: - What the speech commands accepted before the gate

/// A stream runs the local model folder it is given, whatever the language hint: before the gate
/// streams ignored `--language`, so a Parakeet folder with a hint its vocabulary lacks ran
/// Parakeet. Files keep routing the hint to Qwen3-ASR and refuse the folder, as they always did.
@Test func aStreamRunsItsLocalFolderWhateverTheLanguage() throws {
    let root = try temporaryFolder()
    defer { try? FileManager.default.removeItem(at: root) }
    let config = #"{"target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel", "#
        + #""model_defaults": {"tdt_durations": [0, 1, 2, 3, 4]}, "preprocessor": {}, "encoder": {}}"#
    try Data(config.utf8).write(to: root.appendingPathComponent("config.json"))

    let streamed = try SpeechTranscriptionResolver.route(
        task: .transcribe, language: "zz", preferredBackend: .auto, modelOverride: root.path, followsLocalModel: true
    )
    #expect(streamed.decision.backend == .parakeet && streamed.modelOverride == root.path)
    #expect(throws: SpeechTranscriptionIssue.self) {
        try SpeechTranscriptionResolver.route(task: .transcribe, language: "zz", preferredBackend: .auto, modelOverride: root.path)
    }
    let gate = try report("speech", "transcribe", "-", "--stream", "--input-format", "pcm-s16le", "--model", root.path,
                          "--language", "zz")
    #expect(gate.family == "parakeet" && gate.violations.isEmpty, "\(gate)")
}

/// `speech listen` loads Qwen3-ASR whatever `--model` names, so a Parakeet id runs with a
/// warning; `--list-devices` lists inputs before any model is read, on both live commands.
@Test func liveCommandsKeepWhatTheyAcceptedBefore() throws {
    let parakeet = try report("speech", "listen", "--model", "speech-asr-parakeet")
    #expect(parakeet.family == "qwen3-asr" && parakeet.model == "speech-asr-qwen3" && parakeet.violations.isEmpty)
    #expect(parakeet.warnings == [
        "--model speech-asr-parakeet has no effect: speech listen runs Qwen3-ASR. "
            + "Parakeet transcribes recorded audio only; use `speech transcribe`."
    ])
    let alias = try report("speech", "listen", "-m", "mlx-community/parakeet-tdt-0.6b-v3")
    #expect(alias.family == "qwen3-asr" && alias.warnings.count == 1)
    for commandLine in [
        ["speech", "listen", "--list-devices", "--model", "speech-asr-parakeet"],
        ["speech", "diarize-live", "--list-devices", "--model", "speech-diarization-sortformer"],
        ["speech", "diarize-live", "--model", "speech-diarization-sortformer", "--list-devices", "--latency", "0.32"]
    ] {
        let listed = try #require(CLICapabilityGate.evaluate(commandLine: commandLine)).report
        #expect(listed.violations.isEmpty && listed.warnings.isEmpty, "\(commandLine)")
        #expect(throws: Never.self) { try CLICapabilityGate.check(arguments: ["mere.run"] + commandLine) }
    }
    #expect(throws: CLICapabilityGate.Rejection.self) {
        try CLICapabilityGate.check(arguments: ["mere.run", "speech", "diarize-live", "--model", "speech-diarization-sortformer"])
    }
}

/// The value a family always runs passes with a warning, so Studio can hide the control; any
/// other value stays refused.
@Test func aFamilyThatRunsOneValueWarnsOnItAndRefusesTheRest() throws {
    let mlx = try report("speech", "transcribe", "a.wav", "--task", "translate", "--provider", "mlx")
    #expect(mlx.family == "qwen3-asr" && mlx.violations.isEmpty
        && mlx.warnings == ["--provider has no effect with Qwen3-ASR. It applies to Parakeet."])
    let offline = try report("speech", "diarize", "a.wav", "--latency", "offline")
    #expect(offline.family == "sortformer" && offline.violations.isEmpty
        && offline.warnings == ["--latency has no effect with Sortformer. It applies to Nemotron 3 Diarization."])
    #expect(try report("speech", "diarize", "a.wav", "--latency", "1.04").violations.count == 1)
}
