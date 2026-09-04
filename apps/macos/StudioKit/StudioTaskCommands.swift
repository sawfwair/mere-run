import Foundation

extension CommandTemplateID {
    /// The one owning task; secondary entry points can share a command without changing its history identity.
    package var studioTask: StudioTask {
        switch self {
        case .imageGenerate: return .imageGenerate
        case .imageValidate, .imageDatasetDiscover, .imageRunPlan: return .imageDatasets
        case .imageTrainLoRA, .imageVisualizeRun: return .imageTrain
        case .videoGenerate, .videoRetake, .videoDubIt, .videoCosmos3: return .videoGenerate
        case .videoAnimate, .videoPrepareMasks, .videoExportLatents, .videoSession: return .videoSubjects
        case .musicGenerate: return .musicCompose
        case .musicRealtime: return .musicRealtime
        case .musicAnalyze: return .musicAnalyze
        case .musicTranscribe: return .musicTranscribe
        case .musicSeparate: return .musicSeparate
        case .musicTrainAdapter: return .musicTrain
        case .sfxGenerate, .audioGenerate: return .soundGenerate
        case .sfxVideo: return .soundFoley
        case .sfxConditionText: return .soundCondition
        case .sfxAEEncode: return .soundEncode
        case .sfxAEDecode: return .soundDecode
        case .sfxClapScore: return .soundScore
        case .speechSynthesize: return .voiceSpeak
        case .speechProfileList, .speechProfileCreate, .speechProfileDelete: return .voiceVoices
        case .imageReconstruct3D, .imageReconstruct3DTrellis2, .imageReconstruct3DMultiview: return .threeDFromImage
        case .textChat: return .chatChat
        case .textCode: return .chatCode
        case .textTrainLoRA: return .chatTrain
        case .visionInspect, .visionEmbed, .visionCaption, .visionOCR: return .visionRead
        case .visionGround: return .visionFind
        case .visionSegment: return .visionSegment
        case .visionTrack: return .visionTrack
        case .visionDepthVideo: return .visionDepth
        case .visionPose: return .visionPose
        case .visionFaceDetect, .visionFaceEmbed, .visionFaceCompare, .visionFaceBatch: return .visionFaces
        case .visionFlow: return .visionFlow
        case .visionGeometry, .visionGeometryMultiview: return .visionGeometry
        case .visionTrackLive: return .visionLive
        case .speechTranscribe: return .audioTranscribe
        case .speechDiarize: return .audioWhoSpoke
        case .audioEnhance: return .audioEnhance
        case .speechListen: return .audioLive
        case .textEmbed: return .textEmbeddings
        case .textAnonymize: return .textAnonymize
        case .geoFlood: return .earthFlood
        case .geoFire: return .earthFire
        case .geoTessera: return .earthTessera
        case .geoOlmoEarth: return .earthOlmoEarth
        case .modelList, .modelCapabilities, .modelPull, .modelInfo,
             .modelRemove, .modelStorage, .modelRuntimeGet, .modelRuntimeSet: return .modelsInstalled
        case .modelLocationList, .modelLocationAdd, .modelLocationRemove, .modelLocationBind,
             .modelLocationUnbind: return .modelsLocations
        case .modelRepairManifests, .modelOptimize, .modelGarbageCollect, .qualityGate: return .modelsHealth
        case .modelBenchmark, .modelBenchmarkLagunaDFlash, .modelBenchmarkChat, .modelBenchmarkCode,
             .modelBenchmarkFused, .modelBenchmarkFusedFixture, .modelBenchmarkVLM, .modelBenchmarkToolCalls,
             .modelBenchmarkToolContinuations, .modelBenchmarkGemma4KV, .modelBenchmarkGemma4MTP, .modelBenchmarkAPIWorkload: return .modelsBenchmarks
        case .adapterList, .adapterPull: return .modelsAdapters
        case .apiServe, .visionServe, .openWebui, .worldServe,
             .statusSnapshot, .agentOnboard, .agentStatus, .agentInstallPi,
             .agentStart, .setup, .graphStudio, .nodeConsole,
             .custom: return .serverServing
        case .musicServe: return .serverMusic
        case .runList, .runInspect, .runWatch, .runFetch,
             .runCancel, .runRetry, .evaluationPackValidate, .evaluationRun,
             .evaluationPromote: return .runsRuns
        case .pluginList, .pluginInstall, .pluginDoctor, .pluginInfo,
             .pluginRun, .pluginRollback: return .pluginsCatalog
        }
    }
}

extension StudioTask {
    package var commandTemplates: [CommandTemplate] {
        let owner: StudioTask = self == .audioSeparate ? .musicSeparate : (self == .voiceClone ? .voiceSpeak : self)
        return CommandCatalog.templates.filter { $0.id.studioTask == owner && $0.externalURL == nil }
    }
}
