import Foundation
import MereRunContract

/// One field's starting value in a template's draft.
///
/// A `contract` entry reads the CLI's own default out of the shared contract, so Studio never
/// restates a value the CLI already declares. A `value` entry is Studio's own choice, for the
/// options the contract does not describe.
package struct DraftDefault: Sendable {
    /// The flag a `contract` entry reads, for the test that keeps every entry resolvable.
    package let contractFlag: String?
    private let write: @Sendable (inout CommandDraft, MereRunCommandCapability?) -> Void

    package static func value<Value: Sendable>(
        _ keyPath: any WritableKeyPath<CommandDraft, Value> & Sendable, _ value: Value
    ) -> DraftDefault {
        DraftDefault(contractFlag: nil) { draft, _ in
            draft[keyPath: keyPath] = value
        }
    }

    package static func contract<Value: LosslessStringConvertible & Sendable>(
        _ keyPath: any WritableKeyPath<CommandDraft, Value> & Sendable, _ flag: String
    ) -> DraftDefault {
        DraftDefault(contractFlag: flag) { draft, capability in
            guard let declared = capability?.options.first(where: { $0.flag == flag })?.defaultValue,
                  let value = Value(declared) else { return }
            draft[keyPath: keyPath] = value
        }
    }

    package func apply(to draft: inout CommandDraft, capability: MereRunCommandCapability?) {
        write(&draft, capability)
    }
}

/// The starting values each template gives its draft, beyond the prompt, system text, model,
/// and extra arguments the `CommandTemplate` record already carries.
package enum CommandDefaults {
    package static func apply(to draft: inout CommandDraft, id: CommandTemplateID) {
        guard let defaults = byTemplate[id] else { return }
        let capability = id.capability
        for entry in defaults {
            entry.apply(to: &draft, capability: capability)
        }
    }

    /// Split by command family so the type checker never sees one enormous literal.
    package static let byTemplate = Dictionary(
        uniqueKeysWithValues: setup + image + text + speech + vision + media + soundFX + operations
    )

    private static let setup: [(CommandTemplateID, [DraftDefault])] = [
        (.setup, [.value(\.setupMode, "agent"), .value(\.agentModel, "tier")]),
        (.agentStart, [.value(\.port, 8080)])
    ]

    private static let image: [(CommandTemplateID, [DraftDefault])] = [
        (.imageGenerate, [.contract(\.maxSequenceLength, CommandFlags.ImageGenerate.maxSequenceLength)]),
        (.imageTrainLoRA, [.value(\.steps, 1_000), .value(\.maxSequenceLength, 512)]),
        (.imageValidate, [.value(\.backend, "all"), .value(\.variant, "zimage")]),
        (.imageDatasetDiscover, [.value(\.json, true)]),
        (.imageRunPlan, [.value(\.preflight, true), .value(\.json, true)]),
        (.imageVisualizeRun, [.value(\.port, 8787)]),
        (.imageReconstruct3D, [.value(\.reconstructionResolution, 256)]),
        (.imageReconstruct3DTrellis2, [
            .value(\.seed, "42"),
            .value(\.trellisTextureSeed, "42"),
            .value(\.trellisNoRemesh, false),
            .value(\.trellisRemeshBand, 1),
            .value(\.trellisSealRadius, 12),
            .value(\.maxTokens, 2_097_152)
        ]),
        (.imageReconstruct3DMultiview, [.value(\.reconstructionResolution, 128)])
    ]

    private static let text: [(CommandTemplateID, [DraftDefault])] = [
        (.textChat, [.value(\.stream, true)]),
        (.textCode, [
            .contract(\.temperature, CommandFlags.TextCode.temperature),
            .contract(\.topP, CommandFlags.TextCode.topP),
            .value(\.stream, true)
        ]),
        (.textTrainLoRA, [.value(\.steps, 600), .value(\.seed, "42")])
    ]

    private static let speech: [(CommandTemplateID, [DraftDefault])] = [
        (.speechTranscribe, [
            .contract(\.backend, CommandFlags.SpeechTranscribe.backend),
            .contract(\.maxTokens, CommandFlags.SpeechTranscribe.maxTokens),
            .contract(\.task, CommandFlags.SpeechTranscribe.task),
            .value(\.language, "auto"),
            .value(\.timestamps, true)
        ]),
        (.speechDiarize, [
            .value(\.speechDiarizationFormat, "json"),
            .value(\.speechDiarizationThreshold, 0.5),
            .value(\.speechDiarizationMinDuration, 0.25),
            .value(\.speechDiarizationMergeGap, 0.25),
            .value(\.quiet, true)
        ])
    ]

    private static let vision: [(CommandTemplateID, [DraftDefault])] = [
        (.visionEmbed, [.value(\.maxTokens, 8_192)]),
        (.visionCaption, [
            .contract(\.maxTokens, CommandFlags.VisionCaption.maxTokens),
            .contract(\.temperature, CommandFlags.VisionCaption.temperature)
        ]),
        (.visionOCR, [
            .contract(\.maxTokens, CommandFlags.VisionOCR.maxTokens),
            .contract(\.temperature, CommandFlags.VisionOCR.temperature),
            .contract(\.backend, CommandFlags.VisionOCR.backend)
        ]),
        (.visionDepthVideo, [.value(\.dryRun, true)]),
        (.visionGeometry, [.value(\.dryRun, true)]),
        (.visionGeometryMultiview, [.value(\.dryRun, true)]),
        (.visionFaceDetect, [.value(\.json, true)]),
        (.visionFaceEmbed, [.value(\.json, true)]),
        (.visionFaceCompare, [.value(\.json, true)]),
        (.visionFaceBatch, [.value(\.json, true)]),
        (.visionServe, [.value(\.port, 8_091)]),
        (.geoFlood, [.value(\.json, true)]),
        (.geoFire, [.value(\.json, true)]),
        (.geoTessera, [.value(\.json, true)]),
        (.geoOlmoEarth, [.value(\.json, true)])
    ]

    private static let media: [(CommandTemplateID, [DraftDefault])] = [
        (.audioEnhance, [
            .value(\.audioODEMethod, "midpoint"),
            .value(\.audioODESteps, 4),
            .value(\.audioGuidanceScale, 1.5),
            .value(\.audioChunkSeconds, 10),
            .value(\.audioDType, "float32"),
            .value(\.seed, "42")
        ]),
        (.audioGenerate, [
            .value(\.useDuration, true),
            .value(\.durationSeconds, 10),
            .value(\.steps, 30),
            .value(\.seed, "42")
        ]),
        (.musicGenerate, [
            .value(\.steps, 8),
            .value(\.durationSeconds, 10),
            .value(\.useDuration, false),
            .value(\.musicOverrideSteps, false)
        ]),
        (.musicAnalyze, [.value(\.useDuration, false)]),
        (.musicTranscribe, [.value(\.temperature, 1)]),
        (.musicSeparate, [.value(\.audioDType, "float16")]),
        (.musicRealtime, [.value(\.durationSeconds, 30), .value(\.musicPlay, true)]),
        (.musicTrainAdapter, [
            .value(\.steps, 1_000),
            .value(\.rank, 8),
            .value(\.alpha, 16),
            .value(\.learningRate, 0.0001),
            .value(\.seed, "42")
        ]),
        (.musicServe, [.value(\.port, 8081)]),
        (.videoGenerate, [
            .value(\.width, 768),
            .value(\.height, 512),
            .value(\.steps, 40),
            .contract(\.cfgScale, CommandFlags.VideoGenerate.guidanceScale),
            .value(\.videoQuality, .final),
            .value(\.videoOutputMode, .videoOnly)
        ]),
        (.videoRetake, [
            .value(\.steps, 30),
            .value(\.seed, "42"),
            .value(\.retakeStartTime, 0),
            .value(\.retakeEndTime, 4)
        ]),
        (.videoDubIt, [.value(\.width, 768), .value(\.height, 512), .value(\.seed, "42")]),
        (.videoAnimate, [
            .value(\.width, 832),
            .value(\.height, 480),
            .value(\.steps, 40),
            .value(\.cfgScale, 5),
            .value(\.scheduleShift, 3),
            .value(\.fps, 16),
            .value(\.seed, "42")
        ]),
        (.videoCosmos3, [
            .value(\.width, 1280),
            .value(\.height, 720),
            .value(\.numFrames, 189),
            .value(\.steps, 0),
            .value(\.cfgScale, 0),
            .value(\.scheduleShift, 0),
            .value(\.fps, 0),
            .value(\.seed, "0")
        ]),
        (.videoExportLatents, [
            .value(\.width, 768),
            .value(\.height, 512),
            .value(\.numFrames, 65),
            .value(\.seed, "42")
        ]),
        (.worldServe, [.value(\.port, 8791), .value(\.model, "video-dreamx-world-5b-ar-mlx")])
    ]

    private static let soundFX: [(CommandTemplateID, [DraftDefault])] = [
        (.sfxGenerate, [
            .value(\.steps, 4),
            .value(\.durationSeconds, 8),
            .contract(\.cfgScale, CommandFlags.SFXGenerate.cfg)
        ]),
        (.sfxVideo, [.value(\.steps, 4), .value(\.durationSeconds, 8), .value(\.cfgScale, 3)])
    ]

    private static let operations: [(CommandTemplateID, [DraftDefault])] = [
        (.adapterList, [.value(\.json, true)]),
        (.apiServe, [
            .value(\.engine, StudioChatDefaults.fallbackServingEngine),
            .value(\.port, 8080),
            .value(\.contextSize, 32_768)
        ]),
        (.openWebui, [.value(\.host, "0.0.0.0"), .value(\.port, 8080)]),
        (.evaluationPackValidate, [.value(\.json, true)]),
        (.evaluationRun, [.value(\.json, true), .value(\.dryRun, true)]),
        (.evaluationPromote, [.value(\.json, true)]),
        (.modelLocationList, [.value(\.json, true)]),
        (.modelRepairManifests, [.value(\.force, true), .value(\.json, true)]),
        (.modelOptimize, [.value(\.json, true)]),
        (.modelStorage, [.value(\.json, true)]),
        (.modelGarbageCollect, [.value(\.json, true)]),
        (.modelRuntimeGet, [.value(\.json, true)]),
        (.modelRuntimeSet, [.value(\.json, true)]),
        (.modelBenchmark, [
            .value(\.temperature, 0),
            .value(\.topP, 0.9),
            .value(\.contextSize, 16_384)
        ]),
        (.modelBenchmarkLagunaDFlash, [
            .value(\.benchmarkDecodeTokenValues, "8,12,16,24,32,48"),
            .value(\.temperature, 0),
            .value(\.topP, 1),
            .value(\.topK, 0),
            .value(\.minP, 0.02),
            .value(\.contextSize, 4_096),
            .value(\.json, true)
        ]),
        (.modelBenchmarkChat, [.value(\.json, true)]),
        (.modelBenchmarkCode, [.value(\.benchmarkSandbox, "auto"), .value(\.json, true)]),
        (.modelBenchmarkFused, [.value(\.benchmarkSuite, "lite"), .value(\.json, true)]),
        (.modelBenchmarkVLM, [.value(\.benchmarkDataset, "synthetic-vqa-v1"), .value(\.json, true)]),
        (.modelBenchmarkToolCalls, [.value(\.json, true)]),
        (.modelBenchmarkToolContinuations, [.value(\.json, true)]),
        (.modelBenchmarkGemma4KV, [.value(\.json, true)]),
        (.modelBenchmarkGemma4MTP, [.value(\.json, true)]),
        (.modelBenchmarkParakeetCoreML, [
            .value(\.benchmarkWarmupRepetitions, 2),
            .value(\.benchmarkRepetitions, 5),
            .value(\.json, true)
        ]),
        (.modelBenchmarkAPIWorkload, [.value(\.port, 8080), .value(\.json, true)]),
        (.pluginInfo, [.value(\.json, true)]),
        (.runList, [.value(\.json, true)]),
        (.runInspect, [.value(\.json, true)]),
        (.runFetch, [.value(\.json, true)]),
        (.runCancel, [.value(\.json, true)]),
        (.runRetry, [.value(\.json, true)]),
        (.statusSnapshot, [.value(\.json, true)]),
        (.qualityGate, [.value(\.operationsGateSuite, "all")])
    ]
}
