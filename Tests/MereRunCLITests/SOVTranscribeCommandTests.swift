import ArgumentParser
import AudioCore
import AudioSTT
import Foundation
import MereRunCore
import MLX
import XCTest
@testable import MereRunCLI

final class SOVTranscribeCommandTests: XCTestCase {
    func testDecodeProfilingIsExplicitAndRequiresAnUnpairedDraftRun() throws {
        let base = ["audio.wav", "--model-root", "/tmp/sov"]
        XCTAssertFalse(try ModelBenchmarkSOV.parse(base).profileDecode)
        let command = try ModelBenchmarkSOV.parse(base + ["--draft-model", "/tmp/draft", "--profile-decode"])
        XCTAssertTrue(command.profileDecode)
        XCTAssertEqual(command.captureRound, 3)
        for flags in [["--profile-decode"], ["--capture-round", "0"],
                      ["--draft-model", "/tmp/draft", "--profile-decode", "--interleave-metal"],
                      ["--draft-model", "/tmp/draft", "--profile-decode", "--projected-output", "/tmp/x"],
                      ["--draft-model", "/tmp/draft", "--metal-capture", "/tmp/not-a-trace"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testAudioTokenMarginIsExplicitBoundedAndPairsOnOneModel() throws {
        let base = ["audio.wav", "--model-root", "/tmp/sov"]
        XCTAssertNil(try ModelBenchmarkSOV.parse(base).audioTokenMargin)
        for margin in ["0", "12", "25", "375"] {
            let command = try ModelBenchmarkSOV.parse(base + ["--audio-token-margin", margin, "--interleave-audio-tokens", "--audio-tokens-first"])
            XCTAssertEqual(command.audioTokenMargin, Int(margin))
            XCTAssertTrue(command.interleaveAudioTokens)
        }
        for flags in [["--audio-token-margin", "-1"], ["--audio-token-margin", "376"], ["--interleave-audio-tokens"],
                      ["--audio-tokens-first"], ["--audio-token-margin", "25", "--interleave-audio-tokens", "--interleave-generation"],
                      ["--audio-token-margin", "25", "--interleave-audio-tokens", "--projected-output", "/tmp/out"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testPackedWeightsAreExplicitAndScopedToSOV() throws {
        let path = "/Volumes/SSD/sov-q8"
        let command = try SpeechTranscribe.parse(["audio.wav", "--backend", "sov", "--sov-packed-weights", path])
        XCTAssertEqual(command.sovPackedWeights, path)
        XCTAssertNil(try SpeechTranscribe.parse(["audio.wav", "--backend", "sov"]).sovPackedWeights)
        XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--backend", "parakeet", "--sov-packed-weights", path]))
        let report = try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe", "audio.wav", "--backend", "sov", "--sov-packed-weights", path])).report
        XCTAssertEqual(report.violations, [])
        let base = ["audio.wav", "--model-root", "/tmp/sov", "--packed-weights", path]
        XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base))
        XCTAssertEqual(try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8-metal"]).packedWeights, path)
        let options = SOVTranscribeOptions(packedWeights: path)
        XCTAssertEqual(try JSONDecoder().decode(SOVTranscribeOptions.self, from: JSONEncoder().encode(options)), options)
    }

    func testParsesNativeInstructionAndDFlashCompanion() throws {
        let command = try SpeechTranscribe.parse(["audio.wav", "--backend", "sov", "--instruction", "Write numbers as digits.",
                                                 "--draft-model", SOVTranscribeResources.draftModelID])
        XCTAssertEqual(command.backend.backend, .sov)
        XCTAssertEqual(command.instruction, "Write numbers as digits.")
        XCTAssertEqual(command.draftModel, SOVTranscribeResources.draftModelID)
    }

    func testCapabilityGateRoutesSOVAndScopesOptions() throws {
        for selection in [["--backend", "sov"], ["--model", SOVTranscribeResources.modelID], ["--model", SOVTranscribeResources.repository]] {
            let report = try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe", "audio.wav"] + selection
                                                                 + ["--instruction", "Transcribe.", "--draft-model", SOVTranscribeResources.draftModelID])).report
            XCTAssertEqual(report.family, "sov-transcribe")
            XCTAssertEqual(report.violations, [])
        }
        let stream = try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe", "audio.wav", "--backend", "sov", "--stream"])).report
        XCTAssertFalse(stream.violations.isEmpty)
    }

    func testCoreMLEncoderPreservesDFlashSelection() throws {
        let command = try SpeechTranscribe.parse(["audio.wav", "--backend", "sov", "--provider", "coreml",
                                                 "--coreml-encoder", "/tmp/sov-coreml", "--draft-model", SOVTranscribeResources.draftModelID])
        XCTAssertEqual(command.provider, .coreml)
        XCTAssertEqual(command.draftModel, SOVTranscribeResources.draftModelID)
        let report = try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe", "audio.wav", "--backend", "sov",
            "--provider", "coreml", "--coreml-encoder", "/tmp/sov-coreml"])).report
        XCTAssertEqual(report.violations, [])
    }

    func testBenchmarkReceiptsDecodeBeforeAndAfterVerifiedPrefixCounter() throws {
        let fields = #""text":"fixture","tokens":[7],"featureSeconds":0.01,"encoderProjectorSeconds":0.02,"decodeSeconds":0.03,"prefillSeconds":0.01,"generationSeconds":0.02,"cleanupSeconds":0.01,"totalSeconds":0.07,"verificationRounds":3,"acceptedDraftTokens":1,"rejectedRounds":2"#
        let old = try JSONDecoder().decode(SOVMeasuredTranscription.self, from: Data(("{" + fields + "}").utf8))
        XCTAssertNil(old.reusedVerificationRounds)
        let current = try JSONDecoder().decode(SOVMeasuredTranscription.self, from: Data(("{" + fields + #","reusedVerificationRounds":2}"#).utf8))
        XCTAssertEqual(current.reusedVerificationRounds, 2)
    }

    func testBenchmarkPrefillChunkSizeParsesAndRejectsZero() throws {
        let command = try ModelBenchmarkSOV.parse(["audio.wav", "--model-root", "/tmp/sov", "--prefill-chunk-size", "512"])
        XCTAssertEqual(command.prefillChunkSize, 512)
        XCTAssertThrowsError(try ModelBenchmarkSOV.parse(["audio.wav", "--model-root", "/tmp/sov", "--prefill-chunk-size", "0"]))
    }

    func testExperimentalInt8BenchmarkIsExplicitAndRejectsCoreML() throws {
        let command = try ModelBenchmarkSOV.parse(["audio.wav", "--model-root", "/tmp/sov", "--experimental-dynamic-int8"])
        XCTAssertTrue(command.experimentalDynamicInt8)
        XCTAssertFalse(try ModelBenchmarkSOV.parse(["audio.wav", "--model-root", "/tmp/sov"]).experimentalDynamicInt8)
        XCTAssertThrowsError(try ModelBenchmarkSOV.parse(["audio.wav", "--model-root", "/tmp/sov", "--experimental-dynamic-int8", "--artifact", "/tmp/coreml"]))
    }

    func testExperimentalDecoderPolicyRequiresExplicitSelection() throws {
        XCTAssertEqual(try ModelBenchmarkSOV.parse(["a.wav", "--model-root", "/tmp/sov"]).experimentalDecoder.policy, .bf16)
        for policy in SOVDecoderExperts.allCases {
            let flags = ["a.wav", "--model-root", "/tmp/sov", "--experimental-decoder", policy.rawValue]
                + (policy == .mixed ? ["--expert-recipe", "/tmp/recipe.json"] : [])
            let command = try ModelBenchmarkSOV.parse(flags)
            XCTAssertEqual(command.experimentalDecoder.policy, policy)
        }
        XCTAssertThrowsError(try ModelBenchmarkSOV.parse(["a.wav", "--model-root", "/tmp/sov", "--experimental-decoder", "q3"]))
        for flags in [["--experimental-decoder", "mixed"], ["--expert-recipe", "/tmp/recipe.json"],
                      ["--experimental-decoder", "q6", "--expert-recipe", "/tmp/recipe.json"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(["a.wav", "--model-root", "/tmp/sov"] + flags))
        }
    }

    func testExpertComparisonRequiresDistinctWeightsAndExclusiveScheduling() throws {
        let base = ["audio.wav", "--model-root", "/tmp/sov"]
        for policy in ["q4", "q6", "q6-matrix", "mixed"] {
            let flags = ["--experimental-decoder", policy, "--interleave-experts", "--experts-first"]
                + (policy == "mixed" ? ["--expert-recipe", "/tmp/recipe.json"] : [])
            let command = try ModelBenchmarkSOV.parse(base + flags)
            XCTAssertTrue(command.interleaveExperts)
            XCTAssertTrue(command.expertsFirst)
        }
        for flags in [["--interleave-experts"], ["--experts-first"],
                      ["--experimental-decoder", "q6", "--interleave-experts", "--interleave-attention"],
                      ["--experimental-decoder", "q4", "--interleave-experts", "--generation-experiment", "retained-allocator"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testResidencyBudgetIsBoundedAndComparisonIsExclusive() throws {
        let gib = 1_073_741_824
        XCTAssertEqual(SOVBenchmarkResidency.budget(activeBytes: 29 * gib, recommended: 96 * gib), 33 * gib)
        XCTAssertEqual(SOVBenchmarkResidency.budget(activeBytes: 95 * gib, recommended: 96 * gib), 96 * gib - 1_048_576)
        XCTAssertEqual(SOVBenchmarkResidency.budget(activeBytes: gib, recommended: nil), 0)
        let base = ["audio.wav", "--model-root", "/tmp/sov"]
        XCTAssertTrue(try ModelBenchmarkSOV.parse(base + ["--interleave-wired", "--wired-first"]).interleaveWired)
        XCTAssertTrue(try ModelBenchmarkSOV.parse(base + ["--wire-models"]).wireModels)
        for flags in [["--wired-first"], ["--interleave-wired", "--wire-models"],
                      ["--interleave-wired", "--interleave-head-norm"], ["--interleave-wired", "--projected-output", "/tmp/x"],
                      ["--wire-models", "--artifact", "/tmp/artifact"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testNativeResidencyBudgetAppliesAndRestores() async throws {
        guard let output = ProcessInfo.processInfo.environment["MERERUN_SOV_WIRED_RECEIPT"] else {
            throw XCTSkip("Set a receipt path to verify native Metal residency budget application.")
        }
        let recommended = try XCTUnwrap(SOVBenchmarkResidency.recommendedBytes)
        struct Transition: Encodable { let requested: Int; let applied: Int; let restored: Int }
        var transitions: [Transition] = []
        for gib in [33, 53, 78] {
            let requested = min(gib * 1_073_741_824, recommended - 1_048_576)
            let ticket = SOVBenchmarkWiredPolicy().ticket(size: requested)
            let applied = await ticket.start()
            let restored = await ticket.end()
            XCTAssertGreaterThanOrEqual(applied, requested)
            XCTAssertEqual(restored, 0)
            transitions.append(Transition(requested: requested, applied: applied, restored: restored))
        }
        let json = JSONEncoder(); json.outputFormatting = [.prettyPrinted, .sortedKeys]
        try json.encode(transitions).write(to: URL(fileURLWithPath: output))
    }

    func testInterleavedMetalComparisonRequiresOneQ8WeightRepresentation() throws {
        let base = ["audio.wav", "--model-root", "/tmp/sov"]
        let command = try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8", "--interleave-metal", "--metal-first"])
        XCTAssertTrue(command.interleaveMetal)
        XCTAssertTrue(command.metalFirst)
        XCTAssertEqual(command.metalBaseline, .mlx)
        XCTAssertEqual(try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8", "--interleave-metal",
                                                         "--metal-baseline", "reuse"]).metalBaseline, .reuse)
        XCTAssertEqual(try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8", "--interleave-metal",
                                                         "--metal-baseline", "byte"]).metalBaseline, .byte)
        for flags in [["--interleave-metal"], ["--experimental-decoder", "q8-g32", "--interleave-metal"],
                      ["--metal-baseline", "reuse"], ["--metal-baseline", "byte"], ["--metal-baseline", "unknown"],
                      ["--metal-first"], ["--experimental-decoder", "q8", "--interleave-metal", "--projected-output", "/tmp/out"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testExpertFusionIsExplicitAndPairsAgainstTheCurrentLayout() throws {
        let base = ["a.wav", "--model-root", "/tmp/sov"]
        XCTAssertFalse(try ModelBenchmarkSOV.parse(base).experimentalExpertFusion)
        let paired = try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8", "--interleave-metal",
                                                        "--metal-baseline", "layout", "--experimental-expert-fusion"])
        XCTAssertTrue(paired.experimentalExpertFusion)
        XCTAssertEqual(paired.metalBaseline, .layout)
        XCTAssertTrue(try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8-metal",
                                                        "--experimental-expert-fusion"]).experimentalExpertFusion)
        for flags in [["--experimental-expert-fusion"], ["--experimental-decoder", "q8", "--experimental-expert-fusion"],
                      ["--experimental-decoder", "q8", "--interleave-metal", "--metal-baseline", "layout"],
                      ["--metal-baseline", "layout", "--experimental-expert-fusion"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testExpertReductionRequiresExplicitMetalDispatch() throws {
        let base = ["a.wav", "--model-root", "/tmp/sov"]
        XCTAssertFalse(try ModelBenchmarkSOV.parse(base).experimentalExpertReduction)
        XCTAssertFalse(try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8-metal"]).experimentalExpertReduction,
            "Qualified hardware selects reduction without the force flag.")
        XCTAssertFalse(try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8", "--interleave-metal"]).experimentalExpertReduction)
        XCTAssertTrue(try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8-metal",
                                                        "--experimental-expert-reduction"]).experimentalExpertReduction)
        let paired = try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8", "--interleave-metal",
                                                        "--metal-baseline", "layout", "--experimental-expert-reduction"])
        XCTAssertEqual(paired.metalBaseline, .layout)
        XCTAssertTrue(paired.experimentalExpertReduction)
        for flags in [["--experimental-expert-reduction"], ["--experimental-decoder", "q8", "--experimental-expert-reduction"],
                      ["--metal-baseline", "layout", "--experimental-expert-reduction"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testAttentionExperimentsAreExplicitAndPairedModesAreExclusive() throws {
        let base = ["a.wav", "--model-root", "/tmp/sov"]
        XCTAssertFalse(try ModelBenchmarkSOV.parse(base).experimentalAttention)
        XCTAssertTrue(try ModelBenchmarkSOV.parse(base + ["--experimental-attention"]).experimentalAttention)
        let command = try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8-metal", "--interleave-attention", "--attention-first"])
        XCTAssertTrue(command.interleaveAttention)
        XCTAssertTrue(command.attentionFirst)
        for flags in [["--attention-first"], ["--interleave-attention", "--experimental-attention"],
                      ["--interleave-attention", "--experimental-decoder", "q8", "--interleave-metal"],
                      ["--interleave-attention", "--projected-output", "/tmp/output"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testHeadNormalizationExperimentAndPairing() throws {
        let base = ["a.wav", "--model-root", "/tmp/sov"]
        XCTAssertFalse(try ModelBenchmarkSOV.parse(base).experimentalHeadNorm)
        XCTAssertTrue(try ModelBenchmarkSOV.parse(base + ["--experimental-head-norm"]).experimentalHeadNorm)
        let command = try ModelBenchmarkSOV.parse(base + ["--experimental-decoder", "q8-metal", "--experimental-attention",
                                                        "--interleave-head-norm", "--head-norm-first"])
        XCTAssertTrue(command.interleaveHeadNorm)
        XCTAssertTrue(command.headNormFirst)
        for flags in [["--head-norm-first"], ["--interleave-head-norm", "--experimental-head-norm"],
                      ["--interleave-head-norm", "--interleave-attention"],
                      ["--interleave-head-norm", "--experimental-decoder", "q8", "--interleave-metal"],
                      ["--interleave-head-norm", "--projected-output", "/tmp/heads"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
    }

    func testRejectsUnsupportedRequestsBeforeLoading() {
        for flags in [["--stream"], ["--task", "translate"], ["--language", "Japanese"], ["--provider", "coreml"]] {
            XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--backend", "sov"] + flags))
        }
        XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--backend", "qwen", "--instruction", "Transcribe."]))
        XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--backend", "qwen", "--draft-model", SOVTranscribeResources.draftModelID]))
    }

    func testGenerationExperimentPairingKeepsAllocatorStateOutOfPairedComparisons() throws {
        let base = ["a.wav", "--model-root", "/tmp/sov", "--draft-model", "/tmp/draft"]
        XCTAssertEqual(try ModelBenchmarkSOV.parse(base).generationExperiment.policy, .reference)
        for experiment in ["pipeline", "draft-q8"] {
            let command = try ModelBenchmarkSOV.parse(base + ["--generation-experiment", experiment, "--interleave-generation", "--generation-first"])
            XCTAssertTrue(command.interleaveGeneration)
            XCTAssertEqual(command.generationExperiment.rawValue, experiment)
        }
        for flags in [["--interleave-generation"], ["--generation-first"],
                      ["--generation-experiment", "retained-allocator", "--interleave-generation"],
                      ["--generation-experiment", "pipeline", "--interleave-generation", "--interleave-head-norm"],
                      ["--generation-experiment", "pipeline", "--interleave-generation", "--projected-output", "/tmp/x"]] {
            XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + flags))
        }
        XCTAssertThrowsError(try ModelBenchmarkSOV.parse(["a.wav", "--model-root", "/tmp/sov", "--generation-experiment", "draft-q8"]))
    }

    func testRemainingGenerationExperimentsCompareAgainstPipeline() throws {
        let base = ["a.wav", "--model-root", "/tmp/sov", "--draft-model", "/tmp/draft"]
        for mode in ["rollback", "compiled", "block-4", "block-8", "block-12", "adaptive", "adaptive-after-first", "vocabulary-32k"] {
            let command = try ModelBenchmarkSOV.parse(base + ["--generation-experiment", mode,
                "--generation-baseline", "pipeline", "--interleave-generation"])
            XCTAssertEqual(command.generationBaseline.rawValue, "pipeline")
            XCTAssertEqual(command.generationExperiment.rawValue, mode)
        }
        XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + ["--generation-baseline", "rollback"]))
        XCTAssertThrowsError(try ModelBenchmarkSOV.parse(base + ["--generation-experiment", "pipeline", "--generation-baseline", "pipeline", "--interleave-generation"]))
    }

    func testRejectsDFlashAsPrimaryModel() {
        for model in [SOVTranscribeResources.draftModelID, SOVTranscribeResources.draftRepository] {
            XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--backend", "sov", "--model", model]))
        }
    }

    func testOperationDispatchPreservesOptionsAndLanguage() async throws {
        let audio = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try Data().write(to: audio)
        defer { try? FileManager.default.removeItem(at: audio) }
        let options = SOVTranscribeOptions(instruction: "Preserve names.", draftModel: SOVTranscribeResources.draftModelID)
        let result = try await CLIASRRouting.transcribe(request: ASRRequest(audioURL: audio, language: "French", sov: options),
                                                       preferredBackend: .auto, modelOverride: SOVTranscribeResources.modelID, executor: SOVProbe())
        XCTAssertEqual(result.backend, .sov)
        XCTAssertEqual(result.plan.request.sov, options)
        XCTAssertEqual(result.decision.normalizedLanguageHint, "fr")
        XCTAssertEqual(result.result.text, "Native SOV transcription")
    }
}

private struct SOVProbe: SpeechTranscriptionExecutor {
    func transcribeSOV(request: ASRRequest, modelID: String, modelPath: String?,
                       progressHandler: (@Sendable (ASRProgress) -> Void)?) async throws -> ASRResult {
        XCTAssertEqual(modelID, SOVTranscribeResources.modelID)
        XCTAssertEqual(request.sov?.instruction, "Preserve names.")
        return ASRResult(text: "Native SOV transcription", language: "fr")
    }
    func transcribeQwen(request: ASRRequest, modelID: String, modelPath: String?,
                        progressHandler: (@Sendable (ASRProgress) -> Void)?) async throws -> ASRResult {
        XCTFail("Unexpected Qwen dispatch"); return ASRResult(text: "")
    }
    func transcribeParakeet(request: ASRRequest, modelID: String, modelPath: String?,
                            progressHandler: (@Sendable (ASRProgress) -> Void)?) async throws -> ASRResult {
        XCTFail("Unexpected Parakeet dispatch"); return ASRResult(text: "")
    }
}
