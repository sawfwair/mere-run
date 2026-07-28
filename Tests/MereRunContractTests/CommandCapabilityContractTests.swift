import Foundation
import MereRunContract
import Testing

@Test func capabilityCatalogIsStableAndMachineReadable() throws {
    let document = MereRunCapabilityCatalog.document
    #expect(document.schemaVersion == 1)
    #expect(document.commands.map(\.id) == [
        "text.chat",
        "text.code",
        "text.embed",
        "text.anonymize",
        "text.train-lora",
        "image.generate",
        "image.train-lora",
        "image.validate",
        "image.dataset.discover",
        "image.run-plan",
        "image.visualize-run",
        "image.reconstruct-3d",
        "image.reconstruct-3d-trellis2",
        "image.reconstruct-3d-multiview",
        "vision.inspect",
        "vision.caption",
        "vision.ocr",
        "vision.ground",
        "vision.segment",
        "vision.track",
        "vision.track-live",
        "vision.face.detect",
        "vision.face.embed",
        "vision.face.compare",
        "vision.face.batch",
        "vision.pose",
        "vision.flow",
        "vision.depth-video",
        "vision.geometry",
        "vision.geometry-multiview",
        "music.generate",
        "music.analyze",
        "music.transcribe",
        "music.realtime",
        "music.train-adapter",
        "music.serve",
        "video.generate",
        "video.animate",
        "video.cosmos3",
        "video.prepare-masks",
        "video.export-latents",
        "video.session",
        "adapter.list",
        "adapter.pull",
        "run.list",
        "run.inspect",
        "run.watch",
        "run.fetch",
        "run.cancel",
        "run.retry",
        "world.serve",
        "status",
        "gate",
        "model.storage",
        "model.gc",
        "model.runtime.get",
        "model.runtime.set",
        "setup",
        "agent.onboard",
        "agent.install-pi",
        "agent.start",
        "model.list",
        "model.capabilities",
        "model.pull",
        "model.info",
        "model.remove",
        "model.repair-manifests",
        "model.benchmark.q36-mtp",
        "model.benchmark.laguna-dflash",
        "speech.synthesize",
        "speech.transcribe",
        "speech.profile.list",
        "speech.profile.create",
        "speech.profile.delete",
        "sfx.generate",
        "sfx.video.generate",
        "sfx.ae.encode",
        "sfx.ae.decode",
        "sfx.clap.score",
        "sfx.condition.text",
        "plugin.list",
        "plugin.install",
        "plugin.doctor",
        "open-webui.quickstart",
        "api.serve",
        "guide",
        "config.set",
        "config.get",
        "config.unset"
    ])
    #expect(document.commands.count == 89)

    let data = try JSONEncoder().encode(document)
    let decoded = try JSONDecoder().decode(MereRunCapabilityDocument.self, from: data)
    #expect(decoded == document)
}

@Test func capabilityFlagsAreUniqueWithinCommands() {
    for command in MereRunCapabilityCatalog.document.commands {
        let flags = command.options.map(\.flag)
        #expect(Set(flags).count == flags.count, "\(command.id) has duplicate option flags")
    }
}

@Test func textChatChoicesComeFromTypedSharedEnums() {
    let chat = MereRunCapabilityCatalog.textChat
    let responseFormat = chat.options.first { $0.flag == "--response-format" }

    #expect(responseFormat?.choices == TextResponseFormat.allCases.map(\.rawValue))
}

@Test func lagunaControlsAreFirstClassAcrossSharedCommandSurfaces() {
    #expect(MereRunCapabilityCatalog.textChat.options.contains { $0.flag == "--min-p" })
    #expect(MereRunCapabilityCatalog.textCode.options.contains { $0.flag == "--min-p" })
    #expect(
        MereRunCapabilityCatalog.modelRuntimeSet.options
            .first { $0.flag == "--engine" }?
            .choices
            .contains("text-chat-laguna") == true
    )
    #expect(
        MereRunCapabilityCatalog.apiServe.options
            .first { $0.flag == "--engine" }?
            .choices
            .contains("text-chat-laguna") == true
    )
    #expect(
        MereRunCapabilityCatalog.modelBenchmarkLagunaDFlash.command
            == ["model", "benchmark", "laguna-dflash"]
    )
}

@Test func videoProductChoicesComeFromTypedSharedEnums() {
    let generate = MereRunCapabilityCatalog.videoGenerate
    let quality = generate.options.first { $0.flag == "--quality" }
    let outputMode = generate.options.first { $0.flag == "--output-mode" }

    #expect(quality?.choices == LTXVideoQuality.allCases.map(\.rawValue))
    #expect(outputMode?.choices == LTXVideoOutputMode.allCases.map(\.rawValue))
    #expect(!generate.options.contains { $0.flag == "--variant" })
}
