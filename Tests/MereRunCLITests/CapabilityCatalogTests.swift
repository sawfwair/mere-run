import MereRunContract
import Testing

@testable import MereRunCLI

@Test func capabilityFlagsMatchArgumentParserHelp() {
    let helpByID: [String: String] = [
        "text.chat": TextChat.helpMessage(),
        "text.code": TextCode.helpMessage(),
        "text.embed": TextEmbed.helpMessage(),
        "text.anonymize": TextAnonymize.helpMessage(),
        "text.train-lora": TextTrainLoRA.helpMessage(),
        "image.generate": ImageGenerate.helpMessage(),
        "image.train-lora": ImageTrainLoRA.helpMessage(),
        "image.validate": ImageValidate.helpMessage(),
        "image.dataset.discover": ImageDatasetDiscover.helpMessage(),
        "image.run-plan": ImageRunPlan.helpMessage(),
        "image.visualize-run": ImageVisualizeRun.helpMessage(),
        "image.reconstruct-3d": ImageReconstruct3D.helpMessage(),
        "image.reconstruct-3d-trellis2": ImageReconstruct3DTrellis2.helpMessage(),
        "image.reconstruct-3d-multiview": ImageReconstruct3DMultiview.helpMessage(),
        "vision.embed": VisionEmbed.helpMessage(),
        "vision.inspect": VisionInspect.helpMessage(),
        "vision.caption": VisionCaption.helpMessage(),
        "vision.ocr": VisionOCR.helpMessage(),
        "vision.ground": VisionGround.helpMessage(),
        "vision.segment": VisionSegment.helpMessage(),
        "vision.track": VisionTrack.helpMessage(),
        "vision.track-live": VisionTrackLive.helpMessage(),
        "vision.face.detect": VisionFaceDetect.helpMessage(),
        "vision.face.embed": VisionFaceEmbed.helpMessage(),
        "vision.face.compare": VisionFaceCompare.helpMessage(),
        "vision.face.batch": VisionFaceBatch.helpMessage(),
        "vision.pose": VisionPose.helpMessage(),
        "vision.flow": VisionFlow.helpMessage(),
        "vision.depth-video": VisionDepthVideo.helpMessage(),
        "vision.geometry": VisionGeometry.helpMessage(),
        "vision.geometry-multiview": VisionGeometryMultiView.helpMessage(),
        "audio.enhance": AudioEnhance.helpMessage(),
        "audio.generate": AudioGenerate.helpMessage(),
        "music.generate": MusicGenerate.helpMessage(),
        "music.analyze": MusicAnalyze.helpMessage(),
        "music.transcribe": MusicTranscribe.helpMessage(),
        "music.separate": MusicSeparate.helpMessage(),
        "music.realtime": MusicRealtime.helpMessage(),
        "music.train-adapter": MusicTrainAdapter.helpMessage(),
        "music.serve": MusicServe.helpMessage(),
        "video.generate": VideoGenerate.helpMessage(),
        "video.retake": VideoRetake.helpMessage(),
        "video.dub-it": VideoDubIt.helpMessage(),
        "video.animate": VideoAnimate.helpMessage(),
        "video.cosmos3": VideoCosmos3.helpMessage(),
        "video.prepare-masks": VideoPrepareMasks.helpMessage(),
        "video.export-latents": VideoExportLatents.helpMessage(),
        "video.session": VideoSession.helpMessage(),
        "adapter.list": AdapterList.helpMessage(),
        "adapter.pull": AdapterPull.helpMessage(),
        "run.list": RunList.helpMessage(),
        "run.inspect": RunInspect.helpMessage(),
        "run.watch": RunWatch.helpMessage(),
        "run.fetch": RunFetch.helpMessage(),
        "run.cancel": RunCancel.helpMessage(),
        "run.retry": RunRetry.helpMessage(),
        "eval.pack.validate": EvaluationPackValidateCommand.helpMessage(),
        "eval.run": EvaluationRunCommand.helpMessage(),
        "eval.promote": EvaluationPromoteCommand.helpMessage(),
        "world.serve": WorldServe.helpMessage(),
        "status": Status.helpMessage(),
        "gate": Gate.helpMessage(),
        "model.storage": ModelStorage.helpMessage(),
        "model.gc": ModelGarbageCollect.helpMessage(),
        "model.runtime.get": ModelRuntimeGet.helpMessage(),
        "model.runtime.set": ModelRuntimeSet.helpMessage(),
        "setup": Setup.helpMessage(),
        "agent.onboard": AgentOnboard.helpMessage(),
        "agent.status": AgentStatus.helpMessage(),
        "agent.install-pi": AgentInstallPi.helpMessage(),
        "agent.start": AgentStart.helpMessage(),
        "model.list": ModelList.helpMessage(),
        "model.capabilities": ModelCapabilities.helpMessage(),
        "model.pull": ModelPull.helpMessage(),
        "model.info": ModelInfo.helpMessage(),
        "model.remove": ModelRemove.helpMessage(),
        "model.repair-manifests": ModelRepairManifests.helpMessage(),
        "model.optimize": ModelOptimize.helpMessage(),
        "model.benchmark.q36-mtp": ModelBenchmarkQ36MTP.helpMessage(),
        "model.benchmark.laguna-dflash": ModelBenchmarkLagunaDFlash.helpMessage(),
        "speech.synthesize": SpeechSynthesize.helpMessage(),
        "speech.transcribe": SpeechTranscribe.helpMessage(),
        "speech.diarize": SpeechDiarize.helpMessage(),
        "speech.profile.list": SpeechProfileList.helpMessage(),
        "speech.profile.create": SpeechProfileCreate.helpMessage(),
        "speech.profile.delete": SpeechProfileDelete.helpMessage(),
        "sfx.generate": SFXGenerate.helpMessage(),
        "sfx.video.generate": SFXVideoGenerate.helpMessage(),
        "sfx.ae.encode": SFXAEEncode.helpMessage(),
        "sfx.ae.decode": SFXAEDecode.helpMessage(),
        "sfx.clap.score": SFXCLAPScoreCommand.helpMessage(),
        "sfx.condition.text": SFXConditionText.helpMessage(),
        "plugin.list": PluginList.helpMessage(),
        "plugin.install": PluginInstall.helpMessage(),
        "plugin.doctor": PluginDoctor.helpMessage(),
        "open-webui.quickstart": OpenWebUIQuickstart.helpMessage(),
        "api.serve": APIServe.helpMessage(),
        "guide": GuideCommand.helpMessage(),
        "config.set": Config.SetCmd.helpMessage(),
        "config.get": Config.GetCmd.helpMessage(),
        "config.unset": Config.UnsetCmd.helpMessage()
    ]

    for capability in MereRunCapabilityCatalog.document.commands {
        let help = helpByID[capability.id]
        #expect(help != nil, "Missing help fixture for \(capability.id)")
        for option in capability.options {
            #expect(
                help?.contains(option.flag) == true,
                "\(capability.id) advertises \(option.flag), but the CLI help does not"
            )
        }
    }
}

@Test func catalogCommandParsesASelectedCapability() throws {
    let command = try CatalogCommand.parse(["video.generate", "--json"])
    #expect(command.id == "video.generate")
    #expect(command.json)
    #expect(MereRunCapabilityCatalog.command(id: command.id ?? "")?.id == "video.generate")
}
