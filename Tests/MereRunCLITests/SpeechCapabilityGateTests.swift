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
        "--provider coreml is not supported by Qwen3-ASR; it runs mlx. Remove --provider or pass mlx.",
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

/// The contract's routing agrees with the router the command runs, for files and streams alike,
/// on every managed model and every backend, task, and explicit language: the same backend and
/// the same model. The one case left out is the contract's known gap: with neither a backend nor
/// a model, a language Parakeet's router does not recognize sends the CLI to Qwen3-ASR.
@Test func transcribeRoutingAgreesWithTheSpeechResolver() throws {
    let capability = MereRunCapabilityCatalog.speechTranscribe
    let families = try #require(capability.routing).families
    let models: [String?] = [nil] + families.flatMap(\.models)
    for model in models {
        for backend in ASRBackend.allCases {
            for task in [ASRTask.transcribe, .translate] {
                for language in [nil, "en", "zz"] where model != nil || backend != .auto || language != "zz" {
                    var arguments = ["a.wav", "--backend", backend.rawValue, "--task", task.rawValue]
                    if let model { arguments += ["--model", model] }
                    if let language { arguments += ["--language", language] }
                    let context = arguments.joined(separator: " ")
                    let route = try SpeechTranscriptionResolver.route(
                        task: task, language: language, preferredBackend: backend, modelOverride: model
                    )
                    let resolved = capability.resolveFamily(MereRunCommandInvocation(capability: capability, arguments: arguments))
                    guard case let .family(family, resolvedModel, _) = resolved else {
                        Issue.record("\(context) resolved \(resolved)")
                        continue
                    }
                    let parakeet = route.decision.backend == .parakeet
                    let runs = route.modelOverride ?? (parakeet ? ParakeetResources.defaultModelId : Qwen3ASRResources.defaultModelId)
                    #expect(family == (parakeet ? "parakeet" : "qwen3-asr"), "\(context)")
                    #expect(resolvedModel == runs, "\(context)")
                }
            }
        }
    }
}

// MARK: - speech diarize

@Test func diarizeRefusesAStreamingBufferForSortformerIncludingTheDefault() throws {
    let refusal = "--latency 0.64 is not supported by Sortformer; it runs offline. Remove --latency or pass offline."
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

@Test func synthesizeScopesOptionsByMode() throws {
    let style = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--voice", "calm", "--profile", "narrator")
    #expect(style.family == "style" && style.model == "speech-tts-qwen3-nano" && style.violations.isEmpty)
    #expect(style.warnings == ["--profile has no effect with Qwen3-TTS style. It applies to Qwen3-TTS clone."])

    let clone = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--mode", "clone", "-v", "calm",
                           "--ref-audio", "me.wav", "--model", "speech-tts-qwen3-customvoice")
    #expect(clone.family == "clone" && clone.model == "speech-tts-qwen3-customvoice")
    #expect(clone.warnings == ["--voice has no effect with Qwen3-TTS clone. It applies to Qwen3-TTS style."])

    let local = try report("speech", "synthesize", "Hi", "-o", "a.wav", "--mode", "clone", "--model", "/tmp/qwen3-tts")
    #expect(local.family == "clone" && local.model == "/tmp/qwen3-tts" && local.source == .selector)
}

/// Every model the contract lists runs in both modes: the command's own model selection takes
/// it, and the mode alone picks the generator path.
@Test func synthesizeModelsAgreeWithTheCommandsModelSelection() throws {
    let routing = try #require(MereRunCapabilityCatalog.speechSynthesize.routing)
    for family in routing.families {
        for model in family.models {
            #expect(try SpeechSynthesisModelSelection.resolve(model).modelID == model, "\(family.id) \(model)")
        }
        #expect(Set(family.models) == Set(routing.families.flatMap(\.models)), "\(family.id) runs every model")
    }
}
