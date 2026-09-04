import ArgumentParser
import Foundation
import MereRunContract
import Testing

@testable import MereRunCLI

private func capabilityHelpMessages() -> [String: String] {
    [
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
        "config.unset": Config.UnsetCmd.helpMessage(),
        "config.list": Config.ListCmd.helpMessage(),
        "config.path": Config.PathCmd.helpMessage(),
        "geo.flood": GeoFlood.helpMessage(),
        "geo.fire": GeoFire.helpMessage(),
        "geo.tessera": GeoTESSERA.helpMessage(),
        "geo.olmoearth": GeoOlmoEarth.helpMessage(),
        "model.location.list": ModelLocationList.helpMessage(),
        "model.location.add": ModelLocationAdd.helpMessage(),
        "model.location.remove": ModelLocationRemove.helpMessage(),
        "model.location.bind": ModelLocationBind.helpMessage(),
        "model.location.unbind": ModelLocationUnbind.helpMessage(),
        "model.benchmark.chat": ModelBenchmarkChat.helpMessage(),
        "model.benchmark.code": ModelBenchmarkCode.helpMessage(),
        "model.benchmark.fused": ModelBenchmarkFused.helpMessage(),
        "model.benchmark.fused-fixture": ModelBenchmarkFusedFixture.helpMessage(),
        "model.benchmark.vlm": ModelBenchmarkVLM.helpMessage(),
        "model.benchmark.tool-calls": ModelBenchmarkToolCalls.helpMessage(),
        "model.benchmark.tool-continuations": ModelBenchmarkToolContinuations.helpMessage(),
        "model.benchmark.gemma4-kv": ModelBenchmarkGemma4KV.helpMessage(),
        "model.benchmark.gemma4-mtp": ModelBenchmarkGemma4MTP.helpMessage(),
        "model.benchmark.api-workload": ModelBenchmarkAPIWorkload.helpMessage(),
        "plugin.info": PluginInfo.helpMessage(),
        "plugin.run": PluginRun.helpMessage(),
        "plugin.rollback": PluginRollback.helpMessage(),
        "speech.listen": SpeechListen.helpMessage(),
        "vision.serve": VisionServe.helpMessage()
    ]
}

@Test func capabilityFlagsMatchArgumentParserHelp() {
    let helpByID = capabilityHelpMessages()

    for capability in MereRunCapabilityCatalog.document.commands {
        let help = helpByID[capability.id]
        #expect(help != nil, "Missing help fixture for \(capability.id)")
        let declaredFlags = Set(capability.options.map(\.flag))
        let helpFlags = longOptionFlags(in: help ?? "")
        if exactCapabilityOptionIDs.contains(capability.id) {
            #expect(
                declaredFlags == helpFlags,
                "\(capability.id) contract flags \(declaredFlags.sorted()) do not match CLI help \(helpFlags.sorted())"
            )
        } else {
            for option in capability.options {
                #expect(
                    help?.contains(option.flag) == true,
                    "\(capability.id) advertises \(option.flag), but the CLI help does not"
                )
            }
        }
    }
}

/// Newly cataloged capabilities begin with exact option parity. Older entries retain
/// their existing compatibility aliases until those contracts are migrated deliberately.
private let exactCapabilityOptionIDs: Set<String> = [
    "geo.flood", "geo.fire", "geo.tessera", "geo.olmoearth",
    "model.location.list", "model.location.add", "model.location.remove",
    "model.location.bind", "model.location.unbind",
    "model.benchmark.chat", "model.benchmark.code", "model.benchmark.fused",
    "model.benchmark.fused-fixture", "model.benchmark.vlm", "model.benchmark.tool-calls",
    "model.benchmark.tool-continuations", "model.benchmark.gemma4-kv",
    "model.benchmark.gemma4-mtp", "model.benchmark.api-workload",
    "plugin.info", "plugin.run", "plugin.rollback", "speech.listen", "vision.serve"
]

/// A capability that declares an artifact has to name a flag the CLI really
/// parses, so the contract cannot promise shells a destination the command
/// would reject. `adapter pull` and `image run-plan` choose their own paths and
/// declare no flag; `capabilityFileOutputsDeclareADestinationFlag` covers them.
@Test func capabilityOutputFlagsAreAcceptedByTheCLI() {
    let helpByID = capabilityHelpMessages()

    for capability in MereRunCapabilityCatalog.document.commands {
        guard let flag = capability.output.flag else { continue }
        guard let help = helpByID[capability.id] else {
            Issue.record("Missing help fixture for \(capability.id)")
            continue
        }
        let path = capability.command.joined(separator: " ")
        #expect(
            longOptionFlags(in: help).contains(flag),
            "\(capability.id) writes its output to \(flag), which `mere.run \(path) --help` does not advertise"
        )
    }
}

/// Every `default_value` the contract advertises must be the default
/// ArgumentParser renders in `--help`, so a CLI default change that forgets the
/// contract fails here instead of drifting into the shells' forms.
@Test func capabilityDefaultValuesMatchArgumentParserHelp() {
    let helpByID = capabilityHelpMessages()

    for capability in MereRunCapabilityCatalog.document.commands {
        guard let help = helpByID[capability.id] else { continue }
        for option in capability.options {
            guard let defaultValue = option.defaultValue else { continue }
            let block = helpBlock(for: option.flag, in: help)
            #expect(block != nil, "\(capability.id) \(option.flag) is missing from the CLI help")
            // ArgumentParser renders `(default: x)` for plain options and
            // `(values: a, b; default: x)` for enum-backed ones.
            #expect(
                block?.contains("default: \(defaultValue))") == true,
                "\(capability.id) \(option.flag) declares default \(defaultValue) but the CLI help says: \(block ?? "")"
            )
        }
    }
}

/// The help text for one option: its `  --flag ...` line plus the wrapped
/// continuation lines up to the next option or section, re-flowed onto one
/// line so wrapped `(default: …)` clauses can be matched.
private func helpBlock(for flag: String, in help: String) -> String? {
    let lines = help.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    guard let start = lines.firstIndex(where: { line in
        guard line.hasPrefix("  -") else { return false }
        for token in line.trimmingCharacters(in: .whitespaces).split(whereSeparator: \.isWhitespace) {
            guard token.hasPrefix("-") else { break }
            for alias in token.split(whereSeparator: { $0 == "," || $0 == "/" }) {
                let name = alias.prefix { $0 == "-" || $0.isLetter || $0.isNumber }
                if name == flag { return true }
            }
        }
        return false
    }) else { return nil }

    var block = [lines[start]]
    for line in lines[(start + 1)...] {
        guard line.hasPrefix(" "), !line.hasPrefix("  -") else { break }
        block.append(line)
    }
    return block
        .joined(separator: " ")
        .split(whereSeparator: \.isWhitespace)
        .joined(separator: " ")
}

private func longOptionFlags(in help: String) -> Set<String> {
    Set(help.split(separator: "\n").flatMap { line -> [String] in
        guard line.hasPrefix("  -") else { return [] }
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        var flags: [String] = []
        for token in trimmed.split(whereSeparator: \.isWhitespace) {
            guard token.hasPrefix("-") else { break }
            for alias in token.split(separator: ",") where alias.hasPrefix("--") {
                let flag = alias.prefix { character in
                    character == "-" || character.isLetter || character.isNumber
                }
                if flag != "--help" { flags.append(String(flag)) }
            }
        }
        return flags
    })
}

@Test func catalogCommandParsesASelectedCapability() throws {
    let command = try CatalogCommand.parse(["video.generate", "--json"])
    #expect(command.id == "video.generate")
    #expect(command.json)
    #expect(MereRunCapabilityCatalog.command(id: command.id ?? "")?.id == "video.generate")
}

/// Every public CLI leaf command must either be described by the shared
/// capability contract or appear in `contractExemptCommandIDs` with the reason
/// it is deliberately absent. Without this the CLI can grow a command that no
/// shell ever surfaces: `capabilityFlagsMatchArgumentParserHelp` only walks the
/// contract, and the app's inverse coverage test is keyed to the contract too,
/// so an uncataloged command is invisible to both.
let contractExemptCommandIDs: [String: String] = [
    "catalog": "Emits the contract itself; shells compile against MereRunContract instead.",
    "relay.serve": "Relay console owns the control plane. See apps/macos/README.md.",
    "executor.add.ssh": "Relay console owns executor profiles.",
    "executor.add.relay": "Relay console owns executor profiles.",
    "executor.list": "Relay console owns executor profiles.",
    "executor.inspect": "Relay console owns executor profiles.",
    "executor.probe": "Relay console owns executor profiles.",
    "executor.login": "Relay console owns device sign-in.",
    "executor.auth-status": "Relay console owns device sign-in.",
    "executor.logout": "Relay console owns device sign-in.",
    "executor.fleet": "Relay console owns fleet telemetry.",
    "executor.node-refresh": "Relay console owns node lifecycle.",
    "executor.node-configure": "Relay console owns scheduling policy.",
    "executor.remove": "Relay console owns executor profiles.",
    "graph.catalog": "Graph Studio owns workflow authoring.",
    "graph.dataset.discover": "Graph Studio owns workflow authoring.",
    "graph.validate": "Graph Studio owns workflow authoring.",
    "graph.preflight": "Graph Studio owns workflow authoring.",
    "graph.materialize": "Graph Studio owns workflow authoring.",
    "graph.export-job": "Graph Studio owns workflow authoring.",
    "graph.run": "Graph Studio owns workflow execution.",
    "graph.run-job": "Graph Studio owns workflow execution.",
    "graph.submit": "Graph Studio owns workflow execution.",
    "graph.submit-job": "Graph Studio owns workflow execution.",
    "graph.worker.probe": "Machine-to-machine worker protocol, not a shell surface.",
    "graph.worker.execute": "Machine-to-machine worker protocol, not a shell surface.",
    "graph.worker.inspect": "Machine-to-machine worker protocol, not a shell surface.",
    "graph.worker.cancel": "Machine-to-machine worker protocol, not a shell surface.",
    "model.benchmark.q38-verification": "Research-only target-verification microbenchmark, not a product workflow.",
    "vision.image-to-3d": "VFX alias of image.reconstruct-3d, surfaced through the Image workspace.",
    "vision.image-to-3d-trellis2": "VFX alias of image.reconstruct-3d-trellis2.",
    "vision.image-to-3d-multiview": "VFX alias of image.reconstruct-3d-multiview."
]

private func publicLeafCommandIDs() -> [String] {
    var identifiers: [String] = []

    func walk(_ commandTypes: [ParsableCommand.Type], path: [String]) {
        for commandType in commandTypes {
            let configuration = commandType.configuration
            guard configuration.shouldDisplay, let name = configuration.commandName else { continue }
            let next = path + [name]
            if configuration.subcommands.isEmpty {
                identifiers.append(next.joined(separator: "."))
            } else {
                walk(configuration.subcommands, path: next)
            }
        }
    }

    walk(MereRunCLI.configuration.subcommands, path: [])
    return identifiers
}

@Test func everyPublicCLICommandIsCatalogedOrExplicitlyExempt() {
    let cataloged = Set(MereRunCapabilityCatalog.document.commands.map(\.id))
    let leaves = publicLeafCommandIDs()

    for id in leaves {
        #expect(
            cataloged.contains(id) || contractExemptCommandIDs[id] != nil,
            """
            `mere.run \(id.replacingOccurrences(of: ".", with: " "))` is not in the shared \
            capability contract. Add a MereRunCommandCapability for it so shells can surface \
            it, or add it to contractExemptCommandIDs with the reason it stays CLI-only.
            """
        )
    }

    let leafIDs = Set(leaves)
    for (id, reason) in contractExemptCommandIDs {
        #expect(leafIDs.contains(id), "Exemption \(id) no longer matches a CLI command: \(reason)")
        #expect(!cataloged.contains(id), "\(id) is cataloged now; remove its exemption.")
    }
}
