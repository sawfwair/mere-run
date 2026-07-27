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
        "music.generate": MusicGenerate.helpMessage(),
        "music.analyze": MusicAnalyze.helpMessage(),
        "music.transcribe": MusicTranscribe.helpMessage(),
        "music.realtime": MusicRealtime.helpMessage(),
        "music.train-adapter": MusicTrainAdapter.helpMessage(),
        "music.serve": MusicServe.helpMessage(),
        "video.generate": VideoGenerate.helpMessage(),
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
        "world.serve": WorldServe.helpMessage(),
        "status": Status.helpMessage(),
        "gate": Gate.helpMessage(),
        "model.storage": ModelStorage.helpMessage(),
        "model.gc": ModelGarbageCollect.helpMessage(),
        "model.runtime.get": ModelRuntimeGet.helpMessage(),
        "model.runtime.set": ModelRuntimeSet.helpMessage()
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
