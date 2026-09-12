import Foundation

public enum MereRunCapabilityCatalog {
    public static let schemaVersion = 1

    /// The `--progress-json` stderr event shape, one JSON object per line.
    /// `step` is 0-based while a stage is in progress; every determinate stage
    /// ends with exactly one event whose `step == total_steps`, written when
    /// the stage completes or at the end of the run at the latest.
    /// `total_steps == 0` marks an indeterminate stage (token streaming with no
    /// known length) that carries no terminal event.
    public static let progressEventExample =
        #"{"event":"progress","stage":"denoising","step":2,"total_steps":4}"#

    /// The `--receipt` final stdout line. The first output is the primary
    /// artifact; sidecars follow with a `role`. It is printed only after a
    /// successful run, so `exit` is always `0`; `--receipt` is rejected
    /// together with `--preflight`, which produces no result.
    public static let resultReceiptExample =
        #"{"event":"result","exit":0,"outputs":[{"kind":"image","path":"/abs/out.png"}]}"#

    /// Capability ids whose commands print the `--receipt` line.
    public static let receiptCapabilityIDs: [String] = [
        "image.generate", "video.generate", "music.generate", "sfx.generate",
        "speech.synthesize", "speech.transcribe",
        "vision.ground", "vision.segment", "vision.track"
    ]

    /// Capability ids whose commands stream `--progress-json` events.
    public static let progressJSONCapabilityIDs: [String] = [
        "image.generate", "video.generate", "music.generate", "sfx.generate", "speech.synthesize"
    ]

    public static let document = MereRunCapabilityDocument(
        schemaVersion: schemaVersion,
        commands: [
            textChat,
            textCode,
            textEmbed,
            textAnonymize,
            textTrainLoRA,
            imageGenerate,
            imageTrainLoRA,
            imageValidate,
            imageDatasetDiscover,
            imageRunPlan,
            imageVisualizeRun,
            imageReconstruct3D,
            imageReconstruct3DTrellis2,
            imageReconstruct3DMultiview,
            visionEmbed,
            visionInspect,
            visionCaption,
            visionOCR,
            visionGround,
            visionSegment,
            visionTrack,
            visionTrackLive,
            visionFaceDetect,
            visionFaceEmbed,
            visionFaceCompare,
            visionFaceBatch,
            visionPose,
            visionFlow,
            visionDepthVideo,
            visionGeometry,
            visionGeometryMultiview,
            audioEnhance,
            audioGenerate,
            musicGenerate,
            musicAnalyze,
            musicTranscribe,
            musicSeparate,
            musicRealtime,
            musicTrainAdapter,
            musicServe,
            videoGenerate,
            videoRetake,
            videoDubIt,
            videoAnimate,
            videoCosmos3,
            videoPrepareMasks,
            videoExportLatents,
            videoSession,
            adapterList,
            adapterPull,
            runList,
            runInspect,
            runWatch,
            runFetch,
            runCancel,
            runRetry,
            evaluationPackValidate,
            evaluationRun,
            evaluationPromote,
            worldServe,
            visionServe,
            status,
            gate,
            modelStorage,
            modelLocationList,
            modelLocationAdd,
            modelLocationRemove,
            modelLocationBind,
            modelLocationUnbind,
            modelGarbageCollect,
            modelRuntimeGet,
            modelRuntimeSet,
            setup,
            agentOnboard,
            agentStatus,
            agentInstallPi,
            agentStart,
            modelList,
            modelCapabilities,
            modelPull,
            modelInfo,
            modelRemove,
            modelRepairManifests,
            modelOptimize,
            modelBenchmarkQ36MTP,
            modelBenchmarkLagunaDFlash,
            modelBenchmarkParakeetCoreML,
            modelBenchmarkChat,
            modelBenchmarkCode,
            modelBenchmarkFused,
            modelBenchmarkFusedFixture,
            modelBenchmarkVLM,
            modelBenchmarkToolCalls,
            modelBenchmarkToolContinuations,
            modelBenchmarkGemma4KV,
            modelBenchmarkGemma4MTP,
            modelBenchmarkAPIWorkload,
            speechSynthesize,
            speechTranscribe,
            speechDiarize,
            speechProfileList,
            speechProfileCreate,
            speechProfileDelete,
            speechListen,
            sfxGenerate,
            sfxVideoGenerate,
            sfxAEEncode,
            sfxAEDecode,
            sfxCLAPScore,
            sfxConditionText,
            pluginList,
            pluginInfo,
            pluginInstall,
            pluginDoctor,
            pluginRun,
            pluginRollback,
            openWebUIQuickstart,
            apiServe,
            guide,
            configSet,
            configGet,
            configUnset,
            configList,
            configPath,
            geoFlood,
            geoFire,
            geoTessera,
            geoOlmoEarth
        ]
    )

    public static func command(id: String) -> MereRunCommandCapability? {
        document.commands.first { $0.id == id }
    }

}
