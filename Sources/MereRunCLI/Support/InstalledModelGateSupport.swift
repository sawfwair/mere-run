import Foundation
@preconcurrency import MLX
import MediaIO
import MereRunCore

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

private struct InstalledDeepSeekBenchmarkReport: Decodable {
    struct ModelResult: Decodable {
        struct BenchmarkCase: Decodable {
            let tokensGenerated: Int
            let response: String?
            let error: String?
        }

        let model: String
        let engine: String
        let status: String
        let error: String?
        let cases: [BenchmarkCase]
    }

    let models: [ModelResult]
}

private struct InstalledWorldTransitionRequest: Encodable {
    struct Camera: Encodable {
        let motion: String
        let translationMeters: [Int]
        let rotationDegrees: [Int]
    }

    let prompt: String
    let camera: Camera
    let sourceImage: String
    let output: String
    let width: Int
    let height: Int
    let numFrames: Int
    let steps: Int
    let guidanceScale: Int
    let shift: Int
    let seed: Int
    let fps: Int

    enum CodingKeys: String, CodingKey {
        case prompt
        case camera
        case sourceImage
        case output
        case width
        case height
        case numFrames = "num_frames"
        case steps
        case guidanceScale = "guidance_scale"
        case shift
        case seed
        case fps
    }
}

private struct InstalledWorldTransitionAccepted: Decodable {
    let jobID: String

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
    }
}

private struct InstalledWorldJob: Decodable {
    let status: String
    let error: String?
}

private struct InstalledGateJSONDocument: Decodable {
    private struct DynamicCodingKey: CodingKey {
        let stringValue: String
        let intValue: Int?

        init?(stringValue: String) {
            self.stringValue = stringValue
            self.intValue = nil
        }

        init?(intValue: Int) {
            self.stringValue = String(intValue)
            self.intValue = intValue
        }
    }

    init(from decoder: Decoder) throws {
        if let object = try? decoder.container(keyedBy: DynamicCodingKey.self) {
            for key in object.allKeys {
                _ = try object.decode(InstalledGateJSONDocument.self, forKey: key)
            }
            return
        }
        if var array = try? decoder.unkeyedContainer() {
            while !array.isAtEnd {
                _ = try array.decode(InstalledGateJSONDocument.self)
            }
            return
        }

        let value = try decoder.singleValueContainer()
        if value.decodeNil() {
            return
        }
        if (try? value.decode(Bool.self)) != nil
            || (try? value.decode(Double.self)) != nil
            || (try? value.decode(String.self)) != nil {
            return
        }
        throw DecodingError.dataCorruptedError(
            in: value,
            debugDescription: "Unsupported JSON scalar"
        )
    }
}

private struct InstalledDiarizationDocument: Decodable {
    struct Segment: Decodable {
        let speaker: String
        let startSeconds: Double
        let endSeconds: Double

        enum CodingKeys: String, CodingKey {
            case speaker
            case startSeconds = "start_seconds"
            case endSeconds = "end_seconds"
        }
    }

    let speakerCount: Int
    let segments: [Segment]

    enum CodingKeys: String, CodingKey {
        case speakerCount = "speaker_count"
        case segments
    }
}

private struct InstalledMusicSeparationDocument: Decodable {
    struct Model: Decodable {
        let id: String
    }

    struct Stem: Decodable {
        let name: String
        let path: String
        let sha256: String
    }

    let model: Model
    let stems: [Stem]
}

/// One explicit release-smoke recipe per managed runtime kind. The release
/// lane builds checks from the installed inventory, so catalog additions
/// cannot disappear behind a representative "family" check.
struct InstalledModelSmokePlan {
    let check: GateCheck
}

enum InstalledModelSmokePlans {
    static func plan(
        for spec: ManagedModelSpec,
        installedIDs: Set<String>
    ) -> InstalledModelSmokePlan? {
        switch spec.validationKind {
        case .flux2Klein, .bonsaiImage, .zimageTurbo, .hidreamO1, .krea2, .ideogram4SDNQ:
            return direct(spec, route: "image generate") { runner in
                try await runner.installedImageCheck(model: spec.id)
            }

        case .gemma4, .gemma4Unified, .q35, .museGlimmer:
            if spec.category == .visionOCR {
                return direct(spec, route: "vision ocr") { runner in
                    try await runner.installedOCRCheck(model: spec.id)
                }
            }
            if spec.category == .visionChat {
                return direct(spec, route: "text chat --image") { runner in
                    try await runner.installedVisionChatCheck(model: spec.id)
                }
            }
            return direct(spec, route: "text chat") { runner in
                try await runner.installedTextCheck(model: spec.id)
            }

        case .laguna, .lfm2, .inkling, .nemotronH, .codegenGGUF, .hfTextChat:
            return direct(spec, route: "text chat") { runner in
                try await runner.installedTextCheck(model: spec.id)
            }

        case .nemotronOmni:
            return direct(spec, route: "text chat through native Omni runtime") { runner in
                try await runner.installedTextCheck(model: spec.id)
            }

        case .deepseekV4FlashIMatrixGGUF:
            return direct(spec, route: "model benchmark chat via bundled DS4 runtime") { runner in
                try await runner.installedDeepseekCheck(model: spec.id)
            }

        case .gemma4MTPAssistant:
            return companion(
                spec,
                installedIDs: installedIDs,
                candidates: [
                    "text-chat-gemma4-12b-4bit",
                    "text-chat-gemma4-12b",
                ],
                route: "consumed by Gemma 4 text chat"
            ) { runner, primary in
                try await runner.installedTextCheck(model: primary)
            }

        case .q35MTPAssistant:
            return companion(
                spec,
                installedIDs: installedIDs,
                candidates: [
                    Q35Resources.ornith35BMLX4BitModelId,
                    Q35Resources.ornith35BMLX6BitModelId,
                    Q35Resources.ornith35BMLX8BitModelId,
                    Q35Resources.ornith35BMLXModelId,
                ],
                route: "consumed by Ornith 1.5 Qwen-family text chat"
            ) { runner, primary in
                try await runner.installedTextCheck(model: primary)
            }

        case .lagunaDFlash:
            return companion(
                spec,
                installedIDs: installedIDs,
                candidates: ["text-chat-laguna-s-2-1"],
                route: "consumed by Laguna text chat"
            ) { runner, primary in
                try await runner.installedTextCheck(model: primary)
            }

        case .museGlimmerAssistant:
            return companion(
                spec,
                installedIDs: installedIDs,
                candidates: [MuseGlimmerResources.modelId],
                route: "consumed by Muse Glimmer text and vision chat"
            ) { runner, primary in
                try await runner.installedTextCheck(model: primary)
            }

        case .nemotronHDSpark:
            return companion(
                spec,
                installedIDs: installedIDs,
                candidates: [NemotronHResources.modelID],
                route: "consumed by Nemotron 3.5 Lightning text chat"
            ) { runner, primary in
                try await runner.installedTextCheck(model: primary)
            }

        case .qwen3TTS:
            return direct(spec, route: "speech synthesize") { runner in
                try await runner.installedTTSCheck(model: spec.id)
            }

        case .qwen3ASR, .parakeet:
            return companion(
                spec,
                installedIDs: installedIDs,
                candidates: ["speech-tts-qwen3-nano"],
                route: "speech transcribe of generated speech"
            ) { runner, fixtureModel in
                try await runner.installedASRCheck(
                    model: spec.id,
                    fixtureModel: fixtureModel
                )
            }

        case .sortformer:
            return direct(spec, route: "speech diarize of a real A-B-A fixture") { runner in
                try await runner.installedDiarizationCheck(model: spec.id)
            }

        case .qwen3Embedding:
            return direct(spec, route: "text embed") { runner in
                try await runner.installedEmbeddingCheck(model: spec.id)
            }

        case .qwen3VLEmbedding:
            return direct(spec, route: "vision embed with text and image inputs") { runner in
                try await runner.installedMultimodalEmbeddingCheck(model: spec.id)
            }

        case .privacyFilter:
            return direct(spec, route: "text anonymize") { runner in
                try await runner.installedPrivacyCheck(model: spec.id)
            }

        case .lightOnOCR:
            return direct(spec, route: "vision ocr") { runner in
                try await runner.installedOCRCheck(model: spec.id)
            }

        case .sam31:
            return direct(spec, route: "vision segment") { runner in
                try await runner.installedSegmentationCheck(model: spec.id)
            }

        case .falconPerception:
            return direct(spec, route: "vision ground") { runner in
                try await runner.installedGroundingCheck(model: spec.id)
            }

        case .terramindFlood:
            return direct(spec, route: "geo flood normalized tensor smoke") { runner in
                try await runner.installedFloodCheck(model: spec.id)
            }

        case .terramindFire:
            return direct(spec, route: "geo fire normalized tensor smoke") { runner in
                try await runner.installedFireCheck(model: spec.id)
            }

        case .tessera:
            return direct(spec, route: "geo tessera temporal embedding smoke") { runner in
                try await runner.installedTESSERACheck(model: spec.id)
            }

        case .olmoEarth:
            return direct(spec, route: "geo olmoearth multisensor embedding smoke") { runner in
                try await runner.installedOlmoEarthCheck(model: spec.id)
            }

        case .insightFaceBuffaloL:
            return direct(spec, route: "vision face detect") { runner in
                try await runner.installedFaceCheck(model: spec.id)
            }

        case .moge2:
            return direct(spec, route: "vision geometry") { runner in
                try await runner.installedMoGeCheck(model: spec.id)
            }

        case .videoDepthAnything:
            return direct(spec, route: "vision depth-video") { runner in
                try await runner.installedVideoDepthCheck(model: spec.id)
            }

        case .depthAnything3:
            return direct(spec, route: "vision geometry-multiview") { runner in
                try await runner.installedDA3Check(model: spec.id)
            }

        case .tripoSR:
            return direct(spec, route: "vision image-to-3d") { runner in
                try await runner.installedTripoCheck(model: spec.id)
            }

        case .instantMesh:
            return direct(spec, route: "vision image-to-3d-multiview") { runner in
                try await runner.installedInstantMeshCheck(model: spec.id)
            }

        case .trellis2:
            return direct(spec, route: "vision image-to-3d-trellis2") { runner in
                try await runner.installedTrellisCheck(model: spec.id)
            }

        case .aceStep:
            return direct(spec, route: "music generate") { runner in
                try await runner.installedACEStepCheck(model: spec.id)
            }

        case .miniMaxMusic3:
            return direct(spec, route: "music generate") { runner in
                try await runner.installedMiniMaxMusic3Check(model: spec.id)
            }

        case .aceStepLM:
            return companion(
                spec,
                installedIDs: installedIDs,
                candidates: [
                    ModelResolver.ModelID.aceStep.rawValue,
                    ModelResolver.ModelID.aceStepXLTurbo.rawValue,
                    ModelResolver.ModelID.aceStepXLSFT.rawValue,
                    ModelResolver.ModelID.aceStepXLBase.rawValue,
                ],
                route: "music generate with an independent planner"
            ) { runner, primary in
                try await runner.installedACEStepCheck(
                    model: primary,
                    plannerModel: spec.id
                )
            }

        case .magentaRT2:
            return direct(spec, route: "music generate") { runner in
                try await runner.installedMagentaCheck(model: spec.id)
            }

        case .muScriptor:
            return direct(spec, route: "music transcribe") { runner in
                try await runner.installedMusicTranscriptionCheck(model: spec.id)
            }

        case .roFormer:
            return direct(spec, route: "music separate") { runner in
                try await runner.installedMusicSeparationCheck(model: spec.id)
            }

        case .apBWE, .univerSR:
            return direct(spec, route: "audio enhance") { runner in
                try await runner.installedAudioEnhancementCheck(model: spec.id)
            }

        case .woosh:
            if spec.id == "sfx-woosh-vflow-8s" || spec.id == "sfx-woosh-dvflow-8s" {
                return direct(spec, route: "sfx video generate") { runner in
                    try await runner.installedVideoSFXCheck(
                        model: spec.id,
                        synchformerModel: "sfx-woosh-synchformer"
                    )
                }
            }
            return direct(spec, route: "sfx generate") { runner in
                try await runner.installedSFXCheck(model: spec.id)
            }

        case .wooshClap:
            return direct(spec, route: "sfx clap score") { runner in
                try await runner.installedCLAPCheck(model: spec.id)
            }

        case .wooshSynchformer:
            return companion(
                spec,
                installedIDs: installedIDs,
                candidates: [
                    "sfx-woosh-dvflow-8s",
                    "sfx-woosh-vflow-8s",
                ],
                route: "consumed by sfx video generate"
            ) { runner, primary in
                try await runner.installedVideoSFXCheck(
                    model: primary,
                    synchformerModel: spec.id
                )
            }

        case .mmaudio:
            return direct(spec, route: "sfx generate") { runner in
                try await runner.installedSFXCheck(model: spec.id)
            }

        case .ltxVideo:
            return direct(spec, route: "video generate legacy LTX") { runner in
                try await runner.installedLTXCheck(
                    model: spec.id,
                    id: spec.id,
                    arguments: [],
                    requireAudio: false
                )
            }

        case .ltxVideo23MLX:
            return direct(spec, route: "video generate draft") { runner in
                try await runner.installedLTXCheck(
                    model: spec.id,
                    id: spec.id,
                    arguments: ["--quality", "draft", "--output-mode", "video-only"],
                    requireAudio: false
                )
            }

        case .ltxVideo23FullMLX:
            return direct(spec, route: "video generate final audio-video") { runner in
                try await runner.installedLTXCheck(
                    model: spec.id,
                    id: spec.id,
                    arguments: [
                        "--quality", "final",
                        "--output-mode", "audio-video",
                        "--a2v-steps", "4",
                    ],
                    requireAudio: true
                )
            }

        case .ltxVideo23A2VMLX:
            return direct(spec, route: "video generate A2Vid") { runner in
                try await runner.installedA2VidCheck(model: spec.id)
            }

        case .ltxVideo25:
            return direct(spec, route: "video generate LTX 2.5 final audio-video") { runner in
                try await runner.installedLTXCheck(
                    model: spec.id,
                    id: spec.id,
                    arguments: [
                        "--quality", "final",
                        "--output-mode", "audio-video",
                        "--width", "256",
                        "--height", "192",
                        "--num-frames", "25",
                    ],
                    requireAudio: true
                )
            }

        case .wan22TI2VMLX:
            return direct(spec, route: "video generate Wan image-to-video") { runner in
                try await runner.installedWanCheck(model: spec.id)
            }

        case .miniMaxH3MLX:
            return direct(spec, route: "video generate MiniMax-H3 synchronized AV") { runner in
                try await runner.installedMiniMaxH3Check(model: spec.id)
            }

        case .cosmos3EdgeMLX:
            return direct(spec, route: "video cosmos3 image-to-video") { runner in
                try await runner.installedCosmosCheck(model: spec.id)
            }

        case .scail2MLX:
            return direct(spec, route: "video animate SCAIL-2") { runner in
                try await runner.installedSCAILCheck(model: spec.id)
            }

        case .dreamXCausalMLX:
            guard installedIDs.contains("video-wan22-ti2v-5b-mlx") else {
                return nil
            }
            return direct(
                spec,
                requiredModels: [spec.id, "video-wan22-ti2v-5b-mlx"],
                route: "world serve transition"
            ) { runner in
                try await runner.installedDreamXCheck(
                    model: spec.id,
                    baseModel: "video-wan22-ti2v-5b-mlx"
                )
            }
        }
    }

    private static func direct(
        _ spec: ManagedModelSpec,
        requiredModels: [String]? = nil,
        route: String,
        run: @escaping @Sendable (GateRunner) async throws -> GateObservation
    ) -> InstalledModelSmokePlan {
        InstalledModelSmokePlan(check: GateCheck(
            id: "installed-\(spec.id)",
            suite: spec.category.rawValue,
            requiredModels: requiredModels ?? [spec.id],
            comparesBaseline: false,
            successDetail: "direct true inference: \(route)",
            run: run
        ))
    }

    private static func companion(
        _ spec: ManagedModelSpec,
        installedIDs: Set<String>,
        candidates: [String],
        route: String,
        run: @escaping @Sendable (GateRunner, String) async throws -> GateObservation
    ) -> InstalledModelSmokePlan? {
        guard let primary = candidates.first(where: installedIDs.contains) else {
            return nil
        }
        return InstalledModelSmokePlan(check: GateCheck(
            id: "installed-\(spec.id)",
            suite: spec.category.rawValue,
            requiredModels: [spec.id, primary],
            comparesBaseline: false,
            successDetail: "companion consumed by true inference: \(route) via \(primary)"
        ) { runner in
            try await run(runner, primary)
        })
    }

    private static func structural(
        _ spec: ManagedModelSpec,
        route: String,
        run: @escaping @Sendable (GateRunner) async throws -> GateObservation
    ) -> InstalledModelSmokePlan {
        InstalledModelSmokePlan(check: GateCheck(
            id: "installed-\(spec.id)",
            suite: spec.category.rawValue,
            requiredModels: [spec.id],
            comparesBaseline: false,
            successDetail: "structural validation only; native inference is not implemented: \(route)",
            run: run
        ))
    }
}

extension GateRunner {
    func installedModelValidationCheck(model: String) async throws -> GateObservation {
        let run = try await exec(
            ["model", "info", model],
            timeout: 1_800
        )
        guard run.stdout.contains("isValid: true") else {
            throw GateError.invalidArtifact("Managed model validation did not report isValid: true")
        }
        return stdoutObservation(run, label: "model validation")
    }

    func installedTextCheck(model: String) async throws -> GateObservation {
        let run = try await exec(
            [
                "text", "chat",
                "--model", model,
                "--prompt", "Reply with the single word ready.",
                "--max-tokens", "24",
                "--temperature", "0",
                "--thinking",
                "--require-installed",
                "--quiet",
            ],
            timeout: 1_800
        )
        return stdoutObservation(run, label: "text response")
    }

    func installedVisionChatCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let run = try await exec(
            [
                "text", "chat",
                "--model", model,
                "--image", image.path,
                "--prompt", "Name the dominant colored object in this image.",
                "--max-tokens", "24",
                "--temperature", "0",
                "--thinking",
                "--require-installed",
                "--quiet",
            ],
            timeout: 1_800
        )
        return stdoutObservation(run, label: "vision response")
    }

    func installedDeepseekCheck(model: String) async throws -> GateObservation {
        let run = try await exec(
            [
                "model", "benchmark", "chat",
                "--models", model,
                "--cases", "MereChat/1",
                "--max-tokens", "32",
                "--temperature", "0",
                "--log-responses",
                "--json",
            ],
            timeout: 3_600
        )
        let data = Data(run.stdout.utf8)
        let report = try JSONDecoder().decode(InstalledDeepSeekBenchmarkReport.self, from: data)
        guard report.models.count == 1,
              let result = report.models.first,
              result.model == model,
              result.engine == "deepseek-v4-flash-gguf",
              result.status == "completed",
              result.error == nil,
              result.cases.count == 1,
              let benchmarkCase = result.cases.first,
              benchmarkCase.tokensGenerated > 0,
              let response = benchmarkCase.response,
              !response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              benchmarkCase.error == nil else {
            throw GateError.invalidArtifact("DeepSeek benchmark did not produce one completed response")
        }
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: nil
        )
    }

    func installedImageCheck(model: String) async throws -> GateObservation {
        let output = artifactURL(model, extension: "png")
        let run = try await exec(
            [
                "image", "generate",
                "--model", model,
                "--prompt", "A red cube on a clean neutral background.",
                "--width", "512",
                "--height", "512",
                "--steps", "1",
                "--seed", "7",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        let image = try MediaImageIO.decode(output)
        let data = try Data(contentsOf: output)
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: image.width > 0 && image.height > 0 && data.count > 1_024
                ? nil
                : "generated image did not decode"
        )
    }

    func installedTTSCheck(model: String) async throws -> GateObservation {
        let output = artifactURL(model, extension: "wav")
        let run = try await exec(
            [
                "speech", "synthesize",
                "The release smoke is running.",
                "--model", model,
                "--output", output.path,
                "--temperature", "0",
                "--quiet",
            ],
            timeout: 1_800
        )
        return try audioObservation(output, run: run)
    }

    func installedASRCheck(
        model: String,
        fixtureModel: String
    ) async throws -> GateObservation {
        let sentence = "The release smoke proves speech recognition works."
        let input = workDirectory.appendingPathComponent("installed-asr-spoken-source.wav")
        if !FileManager.default.fileExists(atPath: input.path) {
            _ = try await exec(
                [
                    "speech", "synthesize",
                    sentence,
                    "--model", fixtureModel,
                    "--output", input.path,
                    "--temperature", "0",
                    "--quiet",
                ],
                timeout: 1_800
            )
        }
        let backend = model.contains("qwen") ? "qwen" : "parakeet"
        let run = try await exec(
            [
                "speech", "transcribe", input.path,
                "--backend", backend,
                "--model", model,
                "--max-tokens", "16",
                "--no-timestamps",
                "--quiet",
            ],
            timeout: 1_800
        )
        let transcript = run.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        let overlap = Self.wordOverlap(source: sentence, candidate: transcript)
        var combinedTranscript = transcript
        var liveSemanticFailure: String?
        var wallSeconds = run.wallSeconds
        if backend == "parakeet" {
            let liveRun = try await exec(
                [
                    "speech", "transcribe", input.path,
                    "--stream",
                    "--backend", backend,
                    "--model", model,
                    "--no-timestamps",
                    "--quiet",
                ],
                timeout: 1_800
            )
            let liveTranscript = liveRun.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            let liveOverlap = Self.wordOverlap(source: sentence, candidate: liveTranscript)
            combinedTranscript += "\n" + liveTranscript
            wallSeconds += liveRun.wallSeconds
            if liveOverlap < 0.5 {
                liveSemanticFailure = String(
                    format: "Live ASR transcript lost the spoken fixture (%.0f%% word overlap)",
                    liveOverlap * 100
                )
            }
        }
        return GateObservation(
            hash: Self.sha256(Data(combinedTranscript.utf8)),
            secondRunHash: nil,
            wallSeconds: wallSeconds,
            decodeTps: nil,
            semanticFailure: overlap < 0.5
                ? String(
                    format: "ASR transcript lost the spoken fixture (%.0f%% word overlap)",
                    overlap * 100
                )
                : liveSemanticFailure
        )
    }

    func installedDiarizationCheck(model: String) async throws -> GateObservation {
        guard let fixturePath = ProcessInfo.processInfo.environment["MERERUN_SORTFORMER_AUDIO"],
              !fixturePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw GateError.invalidArtifact(
                "MERERUN_SORTFORMER_AUDIO must point to the required real A-B-A speaker fixture"
            )
        }
        let fixture = URL(fileURLWithPath: fixturePath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw GateError.invalidArtifact(
                "MERERUN_SORTFORMER_AUDIO fixture was not found at \(fixture.path)"
            )
        }

        let run = try await exec(
            [
                "speech", "diarize", fixture.path,
                "--model", model,
                "--format", "json",
                "--quiet",
            ],
            timeout: 1_800
        )
        let data = Data(run.stdout.utf8)
        let document = try JSONDecoder().decode(InstalledDiarizationDocument.self, from: data)
        let speakerIDs = document.segments.map(\.speaker)
        let distinctSpeakerIDs = Set(speakerIDs)
        let hasValidSegments = document.segments.allSatisfy { $0.startSeconds >= 0 && $0.endSeconds > $0.startSeconds }
        let firstSpeakerReidentified = speakerIDs.count >= 3
            && speakerIDs.first == speakerIDs.last
            && speakerIDs.dropFirst().dropLast().contains { $0 != speakerIDs.first }
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: document.speakerCount >= 2
                && distinctSpeakerIDs.count >= 2
                && hasValidSegments
                && firstSpeakerReidentified
                ? nil
                : "diarization did not preserve the required A-B-A speaker sequence"
        )
    }

    func installedEmbeddingCheck(model: String) async throws -> GateObservation {
        let output = artifactURL(model, extension: "json")
        let run = try await exec(
            [
                "text", "embed",
                "release smoke",
                "--model", model,
                "--output", output.path,
            ],
            timeout: 900
        )
        let data = try Data(contentsOf: output)
        return GateObservation(
            hash: try Self.embeddingVectorHash(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: nil
        )
    }

    func installedMultimodalEmbeddingCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage(name: "installed-embedding-scene.png")
        let output = artifactURL(model, extension: "json")
        let run = try await exec(
            [
                "vision", "embed",
                "--text", "a red square with a peach circle",
                "--image", image.path,
                "--model", model,
                "--dimensions", "256",
                "--output", output.path,
            ],
            timeout: 1_800
        )
        let data = try Data(contentsOf: output)
        return GateObservation(
            hash: try Self.embeddingVectorHash(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: nil
        )
    }

    func installedPrivacyCheck(model: String) async throws -> GateObservation {
        let run = try await exec(
            [
                "text", "anonymize",
                "Alice Smith can be reached at alice@example.com.",
                "--model", model,
                "--json",
            ],
            timeout: 900
        )
        return jsonStdoutObservation(run, label: "privacy JSON")
    }

    func installedOCRCheck(model: String) async throws -> GateObservation {
        #if canImport(CoreGraphics)
        let image = workDirectory.appendingPathComponent("installed-ocr-page.png")
        _ = GateTextPageRenderer.render(to: image)
        let arguments: [String]
        if model.contains("infinity") {
            arguments = [
                "vision", "ocr", image.path,
                "--backend", "infinity",
                "--infinity-runtime", "native",
                "--infinity-model", model,
                "--max-tokens", "512",
                "--quiet",
            ]
        } else {
            arguments = [
                "vision", "ocr", image.path,
                "--model", model,
                "--max-tokens", "128",
                "--quiet",
            ]
        }
        let run = try await exec(arguments, timeout: 1_800)
        return stdoutObservation(run, label: "OCR output")
        #else
        throw GateError.unsupportedPlatform("OCR fixture rendering requires CoreGraphics")
        #endif
    }

    func installedSegmentationCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let output = artifactURL("\(model)-annotated", extension: "png")
        let json = artifactURL(model, extension: "json")
        let masks = workDirectory.appendingPathComponent("\(safeName(model))-masks", isDirectory: true)
        let run = try await exec(
            [
                "vision", "segment", image.path,
                "--box", "96,96,416,416,cube",
                "--model", model,
                "--output", output.path,
                "--json-output", json.path,
                "--mask-output-dir", masks.path,
                "--resolution", "504",
            ],
            timeout: 1_800
        )
        return try jsonFileObservation(json, run: run, label: "segmentation JSON")
    }

    func installedGroundingCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let output = artifactURL("\(model)-annotated", extension: "png")
        let json = artifactURL(model, extension: "json")
        let run = try await exec(
            [
                "vision", "ground", image.path,
                "--query", "red cube",
                "--model", model,
                "--output", output.path,
                "--json-output", json.path,
            ],
            timeout: 1_800
        )
        return try jsonFileObservation(json, run: run, label: "grounding JSON")
    }

    func installedFloodCheck(model: String) async throws -> GateObservation {
        let input = artifactURL("\(model)-input", extension: "safetensors")
        let output = artifactURL(model, extension: "safetensors")
        try MLX.save(
            arrays: [
                "S2L2A": MLX.zeros([1, 12, 4, 256, 256], dtype: .float32),
                "S1RTC": MLX.zeros([1, 2, 4, 256, 256], dtype: .float32),
                "DEM": MLX.zeros([1, 1, 4, 256, 256], dtype: .float32),
            ],
            metadata: ["format": "mere.run/terramind-flood-smoke-v1"],
            url: input
        )
        let run = try await exec(
            [
                "geo", "flood", input.path,
                "--model", model,
                "--output", output.path,
                "--json",
            ],
            timeout: 1_800
        )
        let data = try Data(contentsOf: output)
        let arrays = try MLX.loadArrays(url: output)
        guard let logits = arrays["logits"] else {
            throw GateError.invalidArtifact("TerraMind flood output did not contain logits")
        }
        let values = logits.asArray(Float.self)
        let valid = logits.shape == [1, 2, 256, 256]
            && values.allSatisfy(\.isFinite)
            && !values.isEmpty
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: valid
                ? nil
                : "TerraMind flood logits were missing, non-finite, or had the wrong shape"
        )
    }

    func installedFireCheck(model: String) async throws -> GateObservation {
        let input = artifactURL("\(model)-input", extension: "safetensors")
        let output = artifactURL(model, extension: "safetensors")
        try MLX.save(
            arrays: [
                "S2L2A": MLX.zeros([1, 12, 4, 256, 256], dtype: .float32),
                "S1RTC": MLX.zeros([1, 2, 4, 256, 256], dtype: .float32),
                "DEM": MLX.zeros([1, 1, 4, 256, 256], dtype: .float32),
            ],
            metadata: ["format": "mere.run/terramind-fire-smoke-v1"],
            url: input
        )
        let run = try await exec(
            ["geo", "fire", input.path, "--model", model, "--output", output.path, "--json"],
            timeout: 1_800
        )
        let data = try Data(contentsOf: output)
        let arrays = try MLX.loadArrays(url: output)
        guard let logits = arrays["logits"] else {
            throw GateError.invalidArtifact("TerraMind fire output did not contain logits")
        }
        let values = logits.asArray(Float.self)
        let valid = logits.shape == [1, 2, 256, 256]
            && values.allSatisfy(\.isFinite)
            && !values.isEmpty
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: valid
                ? nil
                : "TerraMind fire logits were missing, non-finite, or had the wrong shape"
        )
    }

    func installedTESSERACheck(model: String) async throws -> GateObservation {
        let input = artifactURL("\(model)-input", extension: "safetensors")
        let output = artifactURL(model, extension: "safetensors")
        try MLX.save(
            arrays: [
                "S2": MLX.zeros([1, 2, 10], dtype: .float32),
                "S2_DOY": MLXArray([Float(15), 165], [1, 2]),
                "S1_ASC": MLX.zeros([1, 2, 2], dtype: .float32),
                "S1_ASC_DOY": MLXArray([Float(12), 160], [1, 2]),
            ],
            metadata: ["format": "mere.run/tessera-v2-smoke-v1"],
            url: input
        )
        let run = try await exec(
            [
                "geo", "tessera", input.path,
                "--model", model,
                "--dimensions", model.hasSuffix("-teacher") ? "1024" : "128",
                "--output", output.path,
                "--json",
            ],
            timeout: 1_800
        )
        let data = try Data(contentsOf: output)
        let arrays = try MLX.loadArrays(url: output)
        guard let embeddings = arrays["embeddings"] else {
            throw GateError.invalidArtifact("TESSERA output did not contain embeddings")
        }
        let values = embeddings.asArray(Float.self)
        let valid = embeddings.shape == [1, 128]
            && values.allSatisfy(\.isFinite)
            && values.contains { abs($0) > 0.000_001 }
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: valid
                ? nil
                : "TESSERA embeddings were zero, non-finite, or had the wrong shape"
        )
    }

    func installedOlmoEarthCheck(model: String) async throws -> GateObservation {
        guard let source = OlmoEarthResources.spec(for: model) else {
            throw GateError.invalidArtifact("Unsupported OlmoEarth model id: \(model)")
        }
        let input = artifactURL("\(model)-input", extension: "safetensors")
        let output = artifactURL(model, extension: "safetensors")
        try MLX.save(
            arrays: [
                "TIMESTAMPS": MLXArray([Int32(1), 0, 2_026], [1, 1, 3]),
                "S2L2A": MLX.zeros([1, 8, 8, 1, 12], dtype: .float32),
            ],
            metadata: ["format": "mere.run/olmoearth-v1.2-smoke-v1"],
            url: input
        )
        let run = try await exec(
            [
                "geo", "olmoearth", input.path,
                "--model", model,
                "--patch-size", "4",
                "--output", output.path,
                "--json",
            ],
            timeout: 1_800
        )
        let data = try Data(contentsOf: output)
        let arrays = try MLX.loadArrays(url: output)
        guard let embeddings = arrays[OlmoEarthModality.sentinel2L2A.outputTensorName] else {
            throw GateError.invalidArtifact("OlmoEarth output did not contain Sentinel-2 embeddings")
        }
        let values = embeddings.asArray(Float.self)
        let valid = embeddings.shape == [1, 2, 2, source.architecture.embeddingDimension]
            && values.allSatisfy(\.isFinite)
            && values.contains { abs($0) > 0.000_001 }
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: valid
                ? nil
                : "OlmoEarth embeddings were zero, non-finite, or had the wrong shape"
        )
    }

    func installedFaceCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let json = artifactURL(model, extension: "json")
        let run = try await exec(
            [
                "vision", "face", "detect", image.path,
                "--model", model,
                "--score-threshold", "0.1",
                "--json-output", json.path,
                "--json",
            ],
            timeout: 900
        )
        return try jsonFileObservation(json, run: run, label: "face detection JSON")
    }

    func installedMoGeCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let output = workDirectory.appendingPathComponent("\(safeName(model))-geometry", isDirectory: true)
        let run = try await exec(
            [
                "vision", "geometry", image.path,
                "--model", model,
                "--resolution-level", "0",
                "--max-points", "512",
                "--output", output.path,
                "--json",
            ],
            timeout: 1_800
        )
        return try directoryObservation(output, run: run, label: "geometry artifacts")
    }

    func installedVideoDepthCheck(model: String) async throws -> GateObservation {
        let video = try fixtureVideo()
        let output = workDirectory.appendingPathComponent("\(safeName(model))-depth", isDirectory: true)
        let run = try await exec(
            [
                "vision", "depth-video", video.path,
                "--model", model,
                "--input-size", "56",
                "--max-frames", "5",
                "--output", output.path,
                "--json",
            ],
            timeout: 1_800
        )
        return try directoryObservation(output, run: run, label: "video depth artifacts")
    }

    func installedDA3Check(model: String) async throws -> GateObservation {
        let first = try fixtureImage()
        let second = try fixtureImage(name: "scene-second.png", offset: 24)
        let output = workDirectory.appendingPathComponent("\(safeName(model))-scene", isDirectory: true)
        let run = try await exec(
            [
                "vision", "geometry-multiview",
                first.path, second.path,
                "--model", model,
                "--process-resolution", "112",
                "--max-points", "512",
                "--output", output.path,
                "--json",
            ],
            timeout: 1_800
        )
        return try directoryObservation(output, run: run, label: "multiview geometry artifacts")
    }

    func installedTripoCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let output = workDirectory.appendingPathComponent("\(safeName(model))-mesh", isDirectory: true)
        let run = try await exec(
            [
                "vision", "image-to-3d", image.path,
                "--model", model,
                "--resolution", "32",
                "--already-framed",
                "--no-vertex-colors",
                "--output", output.path,
                "--json",
            ],
            timeout: 3_600
        )
        return try directoryObservation(output, run: run, label: "TripoSR mesh artifacts")
    }

    func installedInstantMeshCheck(model: String) async throws -> GateObservation {
        let views = try (0..<4).map {
            try fixtureImage(name: "instant-view-\($0).png", offset: $0 * 12)
        }
        let output = workDirectory.appendingPathComponent("\(safeName(model))-mesh", isDirectory: true)
        var arguments = ["vision", "image-to-3d-multiview"]
        for view in views {
            arguments += ["--view", view.path]
        }
        arguments += [
            "--model", model,
            "--resolution", "32",
            "--no-vertex-colors",
            "--output", output.path,
            "--json",
        ]
        let run = try await exec(arguments, timeout: 3_600)
        return try directoryObservation(output, run: run, label: "InstantMesh artifacts")
    }

    func installedTrellisCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let output = workDirectory.appendingPathComponent("\(safeName(model))-mesh", isDirectory: true)
        let run = try await exec(
            [
                "vision", "image-to-3d-trellis2", image.path,
                "--model", model,
                "--already-framed",
                "--seed", "7",
                "--max-tokens", "2097152",
                "--output", output.path,
                "--json",
            ],
            timeout: 7_200
        )
        return try directoryObservation(output, run: run, label: "TRELLIS.2 mesh artifacts")
    }

    func installedACEStepCheck(
        model: String,
        plannerModel: String? = nil
    ) async throws -> GateObservation {
        let output = artifactURL(model, extension: "wav")
        var arguments = [
            "music", "generate",
            "A short soft electronic pulse.",
            "--model", model,
            "--duration", "2",
            "--quality", "draft",
            "--steps", "1",
            "--candidates", "1",
            "--seed", "7",
            "--no-recipe",
            "--output", output.path,
            "--quiet",
        ]
        if let plannerModel {
            arguments += ["--use-lm", "--lm-model", plannerModel]
        } else {
            arguments.append("--no-lm")
        }
        let run = try await exec(
            arguments,
            timeout: 3_600
        )
        return try audioObservation(output, run: run)
    }

    func installedMagentaCheck(model: String) async throws -> GateObservation {
        let output = artifactURL(model, extension: "wav")
        let run = try await exec(
            [
                "music", "generate",
                "A short soft electronic pulse.",
                "--model", model,
                "--duration", "2",
                "--seed-rotation", "7",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        return try audioObservation(output, run: run)
    }

    func installedMiniMaxMusic3Check(model: String) async throws -> GateObservation {
        let output = artifactURL(model, extension: "wav")
        let run = try await exec(
            [
                "music", "generate",
                "A short soft electronic pulse.",
                "--model", model,
                "--instrumental",
                "--duration", "2",
                "--steps", "1",
                "--seed", "7",
                "--no-recipe",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        return try audioObservation(output, run: run)
    }

    func installedMusicTranscriptionCheck(model: String) async throws -> GateObservation {
        let input = workDirectory.appendingPathComponent("installed-music-source.wav")
        if !FileManager.default.fileExists(atPath: input.path) {
            try Self.writeSineWaveFixture(to: input)
        }
        let output = artifactURL(model, extension: "json")
        let run = try await exec(
            [
                "music", "transcribe", input.path,
                "--model", model,
                "--format", "json",
                "--max-tokens-per-chunk", "64",
                "--chunk-batch-size", "1",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        return try jsonFileObservation(output, run: run, label: "music transcription JSON")
    }

    func installedMusicSeparationCheck(model: String) async throws -> GateObservation {
        let input = workDirectory.appendingPathComponent("installed-music-source.wav")
        if !FileManager.default.fileExists(atPath: input.path) {
            try Self.writeSineWaveFixture(to: input)
        }
        let output = artifactURL(model, extension: "stems")
        let run = try await exec(
            [
                "music", "separate", input.path,
                "--model", model,
                "--output-dir", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        let manifest = output.appendingPathComponent("separation.json")
        let manifestData = try Data(contentsOf: manifest)
        let document = try JSONDecoder().decode(
            InstalledMusicSeparationDocument.self,
            from: manifestData
        )
        let expectedNames = try Self.expectedMusicSeparationStemNames(for: model)
        guard document.model.id == model else {
            throw GateError.invalidArtifact(
                "music separation manifest reported model \(document.model.id), expected \(model)"
            )
        }
        guard document.stems.map(\.name) == expectedNames else {
            throw GateError.invalidArtifact(
                "music separation manifest reported stems \(document.stems.map(\.name)), "
                    + "expected \(expectedNames)"
            )
        }
        var stemPeaks: [Float] = []
        for stem in document.stems {
            let expectedFileName = "\(stem.name).wav"
            guard URL(fileURLWithPath: stem.path).lastPathComponent == expectedFileName else {
                throw GateError.invalidArtifact(
                    "music separation manifest path for \(stem.name) was not \(expectedFileName)"
                )
            }
            let artifact = try audioArtifactObservation(
                output.appendingPathComponent(expectedFileName),
                run: run,
                requireAudible: false
            )
            let observation = artifact.observation
            if let failure = observation.semanticFailure {
                throw GateError.invalidArtifact("\(expectedFileName): \(failure)")
            }
            guard stem.sha256 == observation.hash else {
                throw GateError.invalidArtifact(
                    "music separation manifest hash did not match \(expectedFileName)"
                )
            }
            stemPeaks.append(artifact.peak)
        }
        if let failure = Self.musicSeparationAudibilityFailure(stemPeaks: stemPeaks) {
            throw GateError.invalidArtifact(failure)
        }
        return try jsonFileObservation(manifest, run: run, label: "music separation manifest")
    }

    static func musicSeparationAudibilityFailure(stemPeaks: [Float]) -> String? {
        stemPeaks.contains { $0 > 0.0001 }
            ? nil
            : "music separation produced no audible stems"
    }

    static func expectedMusicSeparationStemNames(for model: String) throws -> [String] {
        switch model {
        case ModelResolver.ModelID.roFormerViperX1297.rawValue:
            ["vocals", "instrumental"]
        case ModelResolver.ModelID.roFormerFourStem.rawValue:
            ["drums", "bass", "other", "vocals"]
        case ModelResolver.ModelID.melRoFormerDereverb.rawValue:
            ["noreverb"]
        case ModelResolver.ModelID.melRoFormerDenoise.rawValue:
            ["dry"]
        default:
            throw GateError.invalidArtifact(
                "unsupported music separation model in installed-model gate: \(model)"
            )
        }
    }

    func installedAudioEnhancementCheck(model: String) async throws -> GateObservation {
        let input = workDirectory.appendingPathComponent("installed-audio-enhancement-source.wav")
        if !FileManager.default.fileExists(atPath: input.path) {
            try Self.writeSineWaveFixture(to: input)
        }
        let output = artifactURL(model, extension: "wav")
        var arguments = [
            "audio", "enhance", input.path,
            "--model", model,
            "--output", output.path,
            "--quiet",
        ]
        if model == ModelResolver.ModelID.univerSRAudio.rawValue {
            arguments += ["--input-rate", "16000"]
        }
        let run = try await exec(arguments, timeout: 3_600)
        return try audioObservation(output, run: run)
    }

    func installedSFXCheck(model: String) async throws -> GateObservation {
        let output = artifactURL(model, extension: "wav")
        let run = try await exec(
            [
                "sfx", "generate",
                "A short wooden click.",
                "--model", model,
                "--duration", "1",
                "--steps", "1",
                "--seed", "7",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        return try audioObservation(output, run: run)
    }

    func installedCLAPCheck(model: String) async throws -> GateObservation {
        let input = workDirectory.appendingPathComponent("installed-clap-source.wav")
        if !FileManager.default.fileExists(atPath: input.path) {
            try Self.writeSineWaveFixture(to: input)
        }
        let run = try await exec(
            [
                "sfx", "clap", "score",
                "A steady electronic tone.",
                input.path,
                "--model", model,
                "--quiet",
            ],
            timeout: 900
        )
        return jsonStdoutObservation(run, label: "CLAP score JSON")
    }

    func installedVideoSFXCheck(
        model: String,
        synchformerModel: String
    ) async throws -> GateObservation {
        let video = try fixtureVideo()
        let output = artifactURL("\(model)-video", extension: "wav")
        let run = try await exec(
            [
                "sfx", "video", "generate",
                "A soft mechanical movement.",
                video.path,
                "--model", model,
                "--synchformer-model", synchformerModel,
                "--duration", "1",
                "--steps", "1",
                "--seed", "7",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        return try audioObservation(output, run: run)
    }

    func installedLTXCheck(
        model: String,
        id: String,
        arguments: [String],
        requireAudio: Bool
    ) async throws -> GateObservation {
        try await videoCheck(
            id: safeName(id),
            model: model,
            arguments: arguments,
            requireAudio: requireAudio
        )
    }

    func installedA2VidCheck(model: String) async throws -> GateObservation {
        let audio = workDirectory.appendingPathComponent("installed-a2vid-source.wav")
        if !FileManager.default.fileExists(atPath: audio.path) {
            try Self.writeSineWaveFixture(to: audio)
        }
        return try await installedLTXCheck(
            model: model,
            id: model,
            arguments: [
                "--audio", audio.path,
                "--a2v-steps", "1",
            ],
            requireAudio: true
        )
    }

    func installedWanCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let output = artifactURL(model, extension: "mp4")
        let run = try await exec(
            [
                "video", "generate",
                "The red cube moves slightly.",
                "--model", model,
                "--image", image.path,
                "--width", "256",
                "--height", "256",
                "--num-frames", "5",
                "--steps", "1",
                "--fps", "8",
                "--seed", "7",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        return try generatedVideoObservation(output, run: run)
    }

    func installedMiniMaxH3Check(model: String) async throws -> GateObservation {
        let output = artifactURL(model, extension: "mp4")
        var arguments = [
            "video", "generate",
            "A red cube rotates slowly on a neutral background with a soft mechanical hum.",
            "--model", model,
            "--width", "128",
            "--height", "128",
            "--num-frames", "22",
            "--steps", "2",
            "--seed", "7",
            "--output", output.path,
            "--quiet",
        ]
        if model == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue {
            let image = try fixtureImage()
            arguments.append(contentsOf: ["--reference", "image:\(image.path)"])
        }
        let run = try await exec(arguments, timeout: 7_200)
        let bytes = try Data(contentsOf: output)
        return GateObservation(
            hash: Self.sha256(bytes),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: try validateVideoArtifact(output, requireAudio: true)
        )
    }

    func installedCosmosCheck(model: String) async throws -> GateObservation {
        let image = try fixtureImage()
        let output = artifactURL(model, extension: "mp4")
        let run = try await exec(
            [
                "video", "cosmos3",
                "The red cube moves slightly.",
                "--mode", "image-to-video",
                "--model", model,
                "--image", image.path,
                "--width", "256",
                "--height", "256",
                "--num-frames", "5",
                "--steps", "1",
                "--fps", "8",
                "--seed", "7",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 3_600
        )
        return try generatedVideoObservation(output, run: run)
    }

    func installedSCAILCheck(model: String) async throws -> GateObservation {
        let fixture = try scailFixture()
        let output = artifactURL(model, extension: "mp4")
        let run = try await exec(
            [
                "video", "animate",
                "A red subject moves across the scene.",
                "--reference", fixture.reference.path,
                "--reference-mask", fixture.referenceMask.path,
                "--driving-video", fixture.drivingVideo.path,
                "--driving-mask", fixture.drivingMask.path,
                "--model", model,
                "--profile", "quality",
                "--width", "256",
                "--height", "256",
                "--steps", "1",
                "--segment-length", "5",
                "--segment-overlap", "1",
                "--tail-policy", "pad-trim",
                "--fps", "8",
                "--seed", "7",
                "--output", output.path,
                "--quiet",
            ],
            timeout: 7_200
        )
        return try generatedVideoObservation(output, run: run)
    }

    func installedDreamXCheck(
        model: String,
        baseModel: String
    ) async throws -> GateObservation {
        let source = try fixtureImage()
        let output = artifactURL(model, extension: "mp4")
        let state = workDirectory.appendingPathComponent("\(safeName(model))-state", isDirectory: true)
        try FileManager.default.createDirectory(at: state, withIntermediateDirectories: true)

        let port = 18_000 + Int.random(in: 0..<1_000)
        let server = Process()
        server.executableURL = executableURL
        server.arguments = [
            "world", "serve",
            "--host", "127.0.0.1",
            "--port", String(port),
            "--base-model", baseModel,
            "--model", model,
            "--state-directory", state.path,
        ]
        server.standardOutput = FileHandle.standardError
        server.standardError = FileHandle.standardError
        let startedAt = Date()
        try server.run()
        defer {
            if server.isRunning {
                server.terminate()
                server.waitUntilExit()
            }
        }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        try await waitForWorldServer(baseURL, process: server)
        let payload = InstalledWorldTransitionRequest(
            prompt: "Move forward past the red cube.",
            camera: .init(
                motion: "forward",
                translationMeters: [0, 0, 1],
                rotationDegrees: [0, 0, 0]
            ),
            sourceImage: source.path,
            output: output.path,
            width: 512,
            height: 320,
            numFrames: 17,
            steps: 1,
            guidanceScale: 5,
            shift: 5,
            seed: 7,
            fps: 8
        )
        let accepted = try await worldRequest(
            baseURL.appendingPathComponent("v1/world/session/transitions"),
            method: "POST",
            body: try JSONEncoder().encode(payload)
        )
        let jobID = try JSONDecoder()
            .decode(InstalledWorldTransitionAccepted.self, from: accepted)
            .jobID

        let deadline = Date().addingTimeInterval(7_200)
        while Date() < deadline {
            try await Task.sleep(nanoseconds: 1_000_000_000)
            let jobData = try await worldRequest(
                baseURL.appendingPathComponent("v1/world/jobs/\(jobID)"),
                method: "GET",
                body: nil
            )
            let job = try JSONDecoder().decode(InstalledWorldJob.self, from: jobData)
            let status = job.status
            if status == "completed" {
                let run = ExecResult(
                    stdout: "",
                    stderr: "",
                    wallSeconds: Date().timeIntervalSince(startedAt)
                )
                return try generatedVideoObservation(output, run: run)
            }
            if status == "failed" || status == "cancelled" {
                throw GateError.invalidArtifact(
                    "world transition \(status): \(job.error ?? "unknown error")"
                )
            }
        }
        throw GateError.timedOut("DreamX world transition")
    }

    private func stdoutObservation(_ run: ExecResult, label: String) -> GateObservation {
        let normalized = run.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return GateObservation(
            hash: Self.sha256(Data(normalized.utf8)),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: normalized.isEmpty ? "\(label) was empty" : nil
        )
    }

    private func jsonStdoutObservation(_ run: ExecResult, label: String) -> GateObservation {
        let data = Data(run.stdout.utf8)
        let valid = (try? JSONDecoder().decode(InstalledGateJSONDocument.self, from: data)) != nil
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: valid ? nil : "\(label) was not valid JSON"
        )
    }

    private func jsonFileObservation(
        _ url: URL,
        run: ExecResult,
        label: String
    ) throws -> GateObservation {
        let data = try Data(contentsOf: url)
        let valid = (try? JSONDecoder().decode(InstalledGateJSONDocument.self, from: data)) != nil
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: valid ? nil : "\(label) was not valid JSON"
        )
    }

    private func audioObservation(_ url: URL, run: ExecResult) throws -> GateObservation {
        try audioArtifactObservation(url, run: run, requireAudible: true).observation
    }

    private func audioArtifactObservation(
        _ url: URL,
        run: ExecResult,
        requireAudible: Bool
    ) throws -> (observation: GateObservation, peak: Float) {
        let data = try Data(contentsOf: url)
        let audio = try MediaAudioIO.decode(url, targetSampleRate: 16_000, channels: 2)
        let peak = audio.samples.map(abs).max() ?? 0
        let semanticFailure: String?
        if audio.samples.isEmpty || data.count <= 1_024 {
            semanticFailure = "generated audio did not decode or was empty"
        } else if requireAudible && peak <= 0.0001 {
            semanticFailure = "generated audio was silent"
        } else {
            semanticFailure = nil
        }
        return (
            GateObservation(
                hash: Self.sha256(data),
                secondRunHash: nil,
                wallSeconds: run.wallSeconds,
                decodeTps: nil,
                semanticFailure: semanticFailure
            ),
            peak
        )
    }

    private func directoryObservation(
        _ url: URL,
        run: ExecResult,
        label: String
    ) throws -> GateObservation {
        let entries = try directoryManifest(url)
        return GateObservation(
            hash: Self.sha256(Data(entries.joined(separator: "\n").utf8)),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: entries.isEmpty ? "\(label) were empty" : nil
        )
    }

    private func generatedVideoObservation(
        _ url: URL,
        run: ExecResult
    ) throws -> GateObservation {
        let data = try Data(contentsOf: url)
        return GateObservation(
            hash: Self.sha256(data),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: try validateVideoArtifact(url, requireAudio: false)
        )
    }

    private func artifactURL(_ model: String, extension pathExtension: String) -> URL {
        workDirectory.appendingPathComponent("\(safeName(model)).\(pathExtension)")
    }

    private func safeName(_ value: String) -> String {
        value.map { character in
            character.isLetter || character.isNumber || character == "-" ? character : "-"
        }.reduce(into: "") { $0.append($1) }
    }

    private func fixtureImage(
        name: String = "installed-scene.png",
        offset: Int = 0
    ) throws -> URL {
        let url = workDirectory.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        let width = 512
        let height = 512
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let pixel = ((y * width) + x) * 4
                rgba[pixel] = 228
                rgba[pixel + 1] = 235
                rgba[pixel + 2] = 242
                rgba[pixel + 3] = 255
                let cubeX = x - offset
                if (112..<400).contains(cubeX), (112..<400).contains(y) {
                    rgba[pixel] = 210
                    rgba[pixel + 1] = 36
                    rgba[pixel + 2] = 48
                }
                let dx = x - (256 + offset)
                let dy = y - 240
                if (dx * dx) + (dy * dy) < 92 * 92 {
                    rgba[pixel] = 242
                    rgba[pixel + 1] = 190
                    rgba[pixel + 2] = 150
                }
            }
        }
        try MediaImageIO.writePNG(
            try MediaImage(width: width, height: height, rgba8: rgba),
            to: url
        )
        return url
    }

    private func fixtureVideo() throws -> URL {
        let url = workDirectory.appendingPathComponent("installed-motion.mp4")
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        let width = 256
        let height = 256
        let frameCount = 9
        var rgb = [UInt8]()
        rgb.reserveCapacity(width * height * 3 * frameCount)
        for frame in 0..<frameCount {
            for y in 0..<height {
                for x in 0..<width {
                    let inside = (48 + frame * 8..<128 + frame * 8).contains(x)
                        && (88..<168).contains(y)
                    rgb += inside ? [210, 36, 48] : [228, 235, 242]
                }
            }
        }
        try MediaVideoIO.writeMP4(
            rgb24: rgb,
            width: width,
            height: height,
            frameCount: frameCount,
            fps: 8,
            to: url
        )
        return url
    }

    private func scailFixture() throws -> (
        reference: URL,
        referenceMask: URL,
        drivingVideo: URL,
        drivingMask: URL
    ) {
        let reference = try fixtureImage(name: "scail-reference.png")
        let referenceMask = workDirectory.appendingPathComponent("scail-reference-mask.png")
        let drivingVideo = workDirectory.appendingPathComponent("scail-driving.mov")
        let drivingMask = workDirectory.appendingPathComponent("scail-driving-mask.mov")

        let width = 256
        let height = 256
        if !FileManager.default.fileExists(atPath: referenceMask.path) {
            var rgba = [UInt8](repeating: 255, count: width * height * 4)
            for y in 64..<192 {
                for x in 64..<192 {
                    let pixel = ((y * width) + x) * 4
                    rgba[pixel] = 0
                    rgba[pixel + 1] = 0
                    rgba[pixel + 2] = 255
                    rgba[pixel + 3] = 255
                }
            }
            try MediaImageIO.writePNG(
                try MediaImage(width: width, height: height, rgba8: rgba),
                to: referenceMask
            )
        }
        if !FileManager.default.fileExists(atPath: drivingVideo.path) {
            var frames: [URL] = []
            for frame in 0..<5 {
                var rgba = [UInt8](repeating: 255, count: width * height * 4)
                for y in 0..<height {
                    for x in 0..<width {
                        let inside = (48 + frame * 8..<112 + frame * 8).contains(x)
                            && (80..<176).contains(y)
                        let pixel = ((y * width) + x) * 4
                        let color: [UInt8] = inside ? [210, 36, 48] : [228, 235, 242]
                        rgba[pixel] = color[0]
                        rgba[pixel + 1] = color[1]
                        rgba[pixel + 2] = color[2]
                        rgba[pixel + 3] = 255
                    }
                }
                let frameURL = workDirectory.appendingPathComponent("scail-driving-\(frame).png")
                try MediaImageIO.writePNG(
                    try MediaImage(width: width, height: height, rgba8: rgba),
                    to: frameURL
                )
                frames.append(frameURL)
            }
            try MediaVideoIO.writePaletteVideo(frameURLs: frames, fps: 8, to: drivingVideo)
        }
        if !FileManager.default.fileExists(atPath: drivingMask.path) {
            var frames: [URL] = []
            for frame in 0..<5 {
                var rgba = [UInt8](repeating: 255, count: width * height * 4)
                for y in 80..<176 {
                    for x in (48 + frame * 8)..<(112 + frame * 8) {
                        let pixel = ((y * width) + x) * 4
                        rgba[pixel] = 0
                        rgba[pixel + 1] = 0
                        rgba[pixel + 2] = 255
                        rgba[pixel + 3] = 255
                    }
                }
                let frameURL = workDirectory.appendingPathComponent("scail-mask-\(frame).png")
                try MediaImageIO.writePNG(
                    try MediaImage(width: width, height: height, rgba8: rgba),
                    to: frameURL
                )
                frames.append(frameURL)
            }
            try MediaVideoIO.writePaletteVideo(frameURLs: frames, fps: 8, to: drivingMask)
        }
        return (reference, referenceMask, drivingVideo, drivingMask)
    }

    private func directoryManifest(_ directory: URL) throws -> [String] {
        guard let enumerator = FileManager.default.enumeratorResolvingSymlinks(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        var entries: [String] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true else { continue }
            let relative = String(url.path.dropFirst(directory.path.count))
            entries.append("\(relative):\(values.fileSize ?? 0)")
        }
        return entries.sorted()
    }

    private func waitForWorldServer(_ baseURL: URL, process: Process) async throws {
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline {
            if !process.isRunning {
                throw GateError.commandFailed(
                    "world serve",
                    exitCode: process.terminationStatus,
                    stderr: "server exited before health check"
                )
            }
            if (try? await worldRequest(
                baseURL.appendingPathComponent("health"),
                method: "GET",
                body: nil
            )) != nil {
                return
            }
            try await Task.sleep(nanoseconds: 250_000_000)
        }
        throw GateError.timedOut("world server health check")
    }

    private func worldRequest(
        _ url: URL,
        method: String,
        body: Data?
    ) async throws -> Data {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body
        if body != nil {
            request.setValue("application/json", forHTTPHeaderField: "content-type")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw GateError.invalidArtifact(
                "world request \(method) \(url.path) returned HTTP \(status): "
                    + String(decoding: data, as: UTF8.self)
            )
        }
        return data
    }
}
