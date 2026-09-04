import Foundation

/// Builds the command line one template runs.
///
/// There is one function per template, grouped into the `Catalog` files by command category.
/// Each starts an `ArgumentBuilder` from the flag constants generated for its capability, so a
/// flag the CLI renames or drops stops compiling here instead of failing a contract test.
package enum CommandArguments {
    package static func build(for id: CommandTemplateID, draft: CommandDraft) -> [String] {
        switch id {
        case .setup: return setup(draft)
        case .agentOnboard: return agentOnboard(draft)
        case .agentStatus: return agentStatus(draft)
        case .agentInstallPi: return agentInstallPi(draft)
        case .agentStart: return agentStart(draft)
        case .modelList: return modelList(draft)
        case .modelCapabilities: return modelCapabilities(draft)
        case .modelPull: return modelPull(draft)
        case .modelInfo: return modelInfo(draft)
        case .modelRemove: return modelRemove(draft)
        case .modelRepairManifests: return modelRepairManifests(draft)
        case .modelOptimize: return modelOptimize(draft)
        case .imageGenerate: return imageGenerate(draft)
        case .imageTrainLoRA: return imageTrainLoRA(draft)
        case .imageValidate: return imageValidate(draft)
        case .imageDatasetDiscover: return imageDatasetDiscover(draft)
        case .imageRunPlan: return imageRunPlan(draft)
        case .imageVisualizeRun: return imageVisualizeRun(draft)
        case .imageReconstruct3D: return imageReconstruct3D(draft)
        case .imageReconstruct3DTrellis2: return imageReconstruct3DTrellis2(draft)
        case .imageReconstruct3DMultiview: return imageReconstruct3DMultiview(draft)
        case .textChat: return textChat(draft)
        case .textCode: return textCode(draft)
        case .textEmbed: return textEmbed(draft)
        case .textAnonymize: return textAnonymize(draft)
        case .textTrainLoRA: return textTrainLoRA(draft)
        case .speechSynthesize: return speechSynthesize(draft)
        case .speechTranscribe: return speechTranscribe(draft)
        case .speechDiarize: return speechDiarize(draft)
        case .speechProfileList: return speechProfileList(draft)
        case .speechProfileCreate: return speechProfileCreate(draft)
        case .speechProfileDelete: return speechProfileDelete(draft)
        case .visionInspect: return visionInspect(draft)
        case .visionEmbed: return visionEmbed(draft)
        case .visionCaption: return visionCaption(draft)
        case .visionOCR: return visionOCR(draft)
        case .visionGround: return visionGround(draft)
        case .visionSegment: return visionSegment(draft)
        case .visionTrack: return visionTrack(draft)
        case .visionTrackLive: return visionTrackLive(draft)
        case .visionFaceDetect: return visionFaceDetect(draft)
        case .visionFaceEmbed: return visionFaceEmbed(draft)
        case .visionFaceCompare: return visionFaceCompare(draft)
        case .visionFaceBatch: return visionFaceBatch(draft)
        case .visionPose: return visionPose(draft)
        case .visionFlow: return visionFlow(draft)
        case .visionDepthVideo: return visionDepthVideo(draft)
        case .visionGeometry: return visionGeometry(draft)
        case .visionGeometryMultiview: return visionGeometryMultiview(draft)
        case .musicGenerate: return musicGenerate(draft)
        case .videoGenerate: return videoGenerate(draft)
        case .videoRetake: return videoRetake(draft)
        case .videoDubIt: return videoDubIt(draft)
        case .videoAnimate: return videoAnimate(draft)
        case .videoCosmos3: return videoCosmos3(draft)
        case .videoPrepareMasks: return videoPrepareMasks(draft)
        case .videoExportLatents: return videoExportLatents(draft)
        case .videoSession: return videoSession(draft)
        case .sfxGenerate: return sfxGenerate(draft)
        case .sfxVideo: return sfxVideo(draft)
        case .audioEnhance: return audioEnhance(draft)
        case .audioGenerate: return audioGenerate(draft)
        case .musicAnalyze: return musicAnalyze(draft)
        case .musicTranscribe: return musicTranscribe(draft)
        case .musicSeparate: return musicSeparate(draft)
        case .musicRealtime: return musicRealtime(draft)
        case .musicTrainAdapter: return musicTrainAdapter(draft)
        case .musicServe: return musicServe(draft)
        case .adapterList: return adapterList(draft)
        case .adapterPull: return adapterPull(draft)
        case .runList: return runList(draft)
        case .runInspect: return runInspect(draft)
        case .runWatch: return runWatch(draft)
        case .runFetch: return runFetch(draft)
        case .runCancel: return runCancel(draft)
        case .runRetry: return runRetry(draft)
        case .evaluationPackValidate: return evaluationPackValidate(draft)
        case .evaluationRun: return evaluationRun(draft)
        case .evaluationPromote: return evaluationPromote(draft)
        case .worldServe: return worldServe(draft)
        case .statusSnapshot: return statusSnapshot(draft)
        case .qualityGate: return qualityGate(draft)
        case .modelStorage: return modelStorage(draft)
        case .modelGarbageCollect: return modelGarbageCollect(draft)
        case .modelRuntimeGet: return modelRuntimeGet(draft)
        case .modelRuntimeSet: return modelRuntimeSet(draft)
        case .graphStudio, .nodeConsole: return externalLauncher()
        case .sfxAEEncode: return sfxAEEncode(draft)
        case .sfxAEDecode: return sfxAEDecode(draft)
        case .sfxClapScore: return sfxClapScore(draft)
        case .sfxConditionText: return sfxConditionText(draft)
        case .modelBenchmark: return modelBenchmark(draft)
        case .modelBenchmarkLagunaDFlash: return modelBenchmarkLagunaDFlash(draft)
        case .pluginList: return pluginList(draft)
        case .pluginInstall: return pluginInstall(draft)
        case .pluginDoctor: return pluginDoctor(draft)
        case .openWebui: return openWebui(draft)
        case .apiServe: return apiServe(draft)
        case .geoFlood: return geoFlood(draft)
        case .geoFire: return geoFire(draft)
        case .geoTessera: return geoTessera(draft)
        case .geoOlmoEarth: return geoOlmoEarth(draft)
        case .modelLocationList: return modelLocationList(draft)
        case .modelLocationAdd: return modelLocationAdd(draft)
        case .modelLocationRemove: return modelLocationRemove(draft)
        case .modelLocationBind: return modelLocationBind(draft)
        case .modelLocationUnbind: return modelLocationUnbind(draft)
        case .modelBenchmarkChat: return modelBenchmarkChat(draft)
        case .modelBenchmarkToolCalls: return modelBenchmarkToolCalls(draft)
        case .modelBenchmarkCode: return modelBenchmarkCode(draft)
        case .modelBenchmarkFused: return modelBenchmarkFused(draft)
        case .modelBenchmarkFusedFixture: return modelBenchmarkFusedFixture(draft)
        case .modelBenchmarkVLM: return modelBenchmarkVLM(draft)
        case .modelBenchmarkToolContinuations: return modelBenchmarkToolContinuations(draft)
        case .modelBenchmarkGemma4KV: return modelBenchmarkGemma4KV(draft)
        case .modelBenchmarkGemma4MTP: return modelBenchmarkGemma4MTP(draft)
        case .modelBenchmarkAPIWorkload: return modelBenchmarkAPIWorkload(draft)
        case .pluginInfo: return pluginInfo(draft)
        case .pluginRun: return pluginRun(draft)
        case .pluginRollback: return pluginRollback(draft)
        case .speechListen: return speechListen(draft)
        case .visionServe: return visionServe(draft)
        case .custom: return custom(draft)
        }
    }
}

// MARK: - Shared value rendering

extension CommandArguments {
    /// Renders a numeric control the way the CLI's parsers read it back: short enough to stay
    /// readable in the command preview, precise enough to round-trip a slider's value.
    package static func format(_ value: Double) -> String {
        String(format: "%.4g", value)
    }

    /// The non-empty lines of a multi-line text field.
    package static func lineList(_ raw: String) -> [String] {
        raw.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    /// The paths in a field that accepts one per line or several separated by commas.
    package static func pathList(_ raw: String) -> [String] {
        raw.components(separatedBy: .newlines)
            .flatMap { $0.split(separator: ",", omittingEmptySubsequences: true) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    /// The options every face command shares. The contract declares them once, so the
    /// constants are read off one of the four capabilities that use the shared list.
    package static func appendFaceOptions(to args: inout ArgumentBuilder, draft: CommandDraft) {
        typealias F = CommandFlags.VisionFaceDetect
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.scoreThreshold, format(draft.visionFaceScoreThreshold))
        args.option(F.executionProvider, draft.visionExecutionProvider)
        if !draft.visionJSONOutputPath.isBlank {
            args.option(F.jsonOutput, draft.visionJSONOutputPath)
        }
        if draft.json { args.flag(F.json) }
    }
}
