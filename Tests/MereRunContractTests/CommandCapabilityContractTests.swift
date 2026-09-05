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
        "vision.embed",
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
        "audio.enhance",
        "audio.generate",
        "music.generate",
        "music.analyze",
        "music.transcribe",
        "music.separate",
        "music.realtime",
        "music.train-adapter",
        "music.serve",
        "video.generate",
        "video.retake",
        "video.dub-it",
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
        "eval.pack.validate",
        "eval.run",
        "eval.promote",
        "world.serve",
        "vision.serve",
        "status",
        "gate",
        "model.storage",
        "model.location.list",
        "model.location.add",
        "model.location.remove",
        "model.location.bind",
        "model.location.unbind",
        "model.gc",
        "model.runtime.get",
        "model.runtime.set",
        "setup",
        "agent.onboard",
        "agent.status",
        "agent.install-pi",
        "agent.start",
        "model.list",
        "model.capabilities",
        "model.pull",
        "model.info",
        "model.remove",
        "model.repair-manifests",
        "model.optimize",
        "model.benchmark.q36-mtp",
        "model.benchmark.laguna-dflash",
        "model.benchmark.parakeet-coreml",
        "model.benchmark.chat",
        "model.benchmark.code",
        "model.benchmark.fused",
        "model.benchmark.fused-fixture",
        "model.benchmark.vlm",
        "model.benchmark.tool-calls",
        "model.benchmark.tool-continuations",
        "model.benchmark.gemma4-kv",
        "model.benchmark.gemma4-mtp",
        "model.benchmark.api-workload",
        "speech.synthesize",
        "speech.transcribe",
        "speech.diarize",
        "speech.profile.list",
        "speech.profile.create",
        "speech.profile.delete",
        "speech.listen",
        "sfx.generate",
        "sfx.video.generate",
        "sfx.ae.encode",
        "sfx.ae.decode",
        "sfx.clap.score",
        "sfx.condition.text",
        "plugin.list",
        "plugin.info",
        "plugin.install",
        "plugin.doctor",
        "plugin.run",
        "plugin.rollback",
        "open-webui.quickstart",
        "api.serve",
        "guide",
        "config.set",
        "config.get",
        "config.unset",
        "config.list",
        "config.path",
        "geo.flood",
        "geo.fire",
        "geo.tessera",
        "geo.olmoearth"
    ])
    #expect(document.commands.count == 128)

    let data = try JSONEncoder().encode(document)
    let decoded = try JSONDecoder().decode(MereRunCapabilityDocument.self, from: data)
    #expect(decoded == document)
}

/// The capabilities behind Studio's prompt modes. Every option on these carries
/// the group and tier metadata the contract-driven inspector renders from.
private let promptModeCapabilityIDs: [String] = [
    "image.generate", "text.chat", "text.code",
    "speech.synthesize", "speech.transcribe",
    "vision.inspect", "vision.ocr", "vision.caption", "vision.ground", "vision.segment", "vision.track",
    "music.generate", "video.generate", "sfx.generate"
]

private let knownOptionGroups: Set<String> = [
    MereRunCapabilityOptionGroup.prompt,
    MereRunCapabilityOptionGroup.inputs,
    MereRunCapabilityOptionGroup.output,
    MereRunCapabilityOptionGroup.modelAndAdapters,
    MereRunCapabilityOptionGroup.sampling,
    MereRunCapabilityOptionGroup.run
]

@Test func promptModeCapabilitiesDeclareOptionMetadata() {
    for id in promptModeCapabilityIDs {
        let capability = MereRunCapabilityCatalog.command(id: id)
        #expect(capability != nil, "Missing prompt-mode capability \(id)")
        for option in capability?.options ?? [] {
            #expect(option.group != nil, "\(id) \(option.flag) has no group")
            #expect(option.tier != nil, "\(id) \(option.flag) has no tier")
        }
        #expect(
            capability?.options.contains { $0.tier == .essential } == true,
            "\(id) declares no essential option for the chip strip"
        )
    }
}

@Test func capabilityOptionMetadataIsWellFormed() {
    for command in MereRunCapabilityCatalog.document.commands {
        let flags = Set(command.options.map(\.flag))
        for option in command.options {
            let context = "\(command.id) \(option.flag)"

            if let group = option.group {
                #expect(knownOptionGroups.contains(group), "\(context) uses unknown group \(group)")
            }

            if let value = option.defaultValue {
                switch option.kind {
                case .integer:
                    #expect(Int(value) != nil, "\(context) default \(value) is not an integer")
                case .number:
                    #expect(Double(value) != nil, "\(context) default \(value) is not a number")
                case .boolean:
                    #expect(["true", "false"].contains(value), "\(context) default \(value) is not a boolean")
                case .choice:
                    #expect(option.choices.contains(value), "\(context) default \(value) is not one of \(option.choices)")
                case .string, .file, .directory:
                    #expect(!value.isEmpty, "\(context) declares an empty default")
                }
            }

            if let dependsOn = option.dependsOn {
                #expect(dependsOn != option.flag, "\(context) depends on itself")
                #expect(flags.contains(dependsOn), "\(context) depends on undeclared flag \(dependsOn)")
            }

            guard let range = option.range else { continue }
            #expect(option.kind == .integer || option.kind == .number, "\(context) declares a range on a non-numeric option")
            #expect(range.min != nil || range.max != nil || range.step != nil, "\(context) declares an empty range")
            if let min = range.min, let max = range.max {
                #expect(min <= max, "\(context) range min \(min) exceeds max \(max)")
            }
            if let step = range.step {
                #expect(step > 0, "\(context) range step must be positive")
            }
            if option.kind == .integer {
                for bound in [range.min, range.max, range.step].compactMap({ $0 }) {
                    #expect(bound == bound.rounded(), "\(context) integer range uses fractional bound \(bound)")
                }
            }
            if let value = option.defaultValue, let number = Double(value) {
                if let min = range.min {
                    #expect(number >= min, "\(context) default \(value) is below range min \(min)")
                }
                if let max = range.max {
                    #expect(number <= max, "\(context) default \(value) is above range max \(max)")
                }
            }
        }
    }
}

@Test func receiptAndProgressFlagsAreDeclaredExactlyWhereEmitted() {
    let commands = MereRunCapabilityCatalog.document.commands
    let withReceipt = commands.filter { $0.options.contains { $0.flag == "--receipt" } }.map(\.id)
    let withProgress = commands.filter { $0.options.contains { $0.flag == "--progress-json" } }.map(\.id)

    #expect(Set(withReceipt) == Set(MereRunCapabilityCatalog.receiptCapabilityIDs))
    #expect(Set(withProgress) == Set(MereRunCapabilityCatalog.progressJSONCapabilityIDs))
    #expect(Set(MereRunCapabilityCatalog.progressJSONCapabilityIDs).isSubset(of: Set(withReceipt)))
}

@Test func optionMetadataSerializesAdditivelyAndDecodesLegacyJSON() throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]

    let width = try #require(MereRunCapabilityCatalog.imageGenerate.options.first { $0.flag == "--width" })
    let widthJSON = String(decoding: try encoder.encode(width), as: UTF8.self)
    #expect(widthJSON.contains(#""default_value":"1024""#))
    #expect(widthJSON.contains(#""group":"Output""#))
    #expect(widthJSON.contains(#""tier":"essential""#))
    #expect(widthJSON.contains(#""range":{"max":2048,"min":256,"step":16}"#))
    #expect(!widthJSON.contains("depends_on"))

    let mask = try #require(MereRunCapabilityCatalog.imageGenerate.options.first { $0.flag == "--mask" })
    #expect(String(decoding: try encoder.encode(mask), as: UTF8.self).contains(#""depends_on":"--input""#))

    let bare = MereRunCapabilityOption(flag: "--x", label: "X", kind: .string)
    let bareJSON = String(decoding: try encoder.encode(bare), as: UTF8.self)
    #expect(bareJSON == #"{"choices":[],"flag":"--x","kind":"string","label":"X","repeatable":false,"required":false}"#)

    let legacy = Data(#"{"flag":"--y","label":"Y","kind":"integer","required":false,"repeatable":false,"choices":[]}"#.utf8)
    let decoded = try JSONDecoder().decode(MereRunCapabilityOption.self, from: legacy)
    #expect(decoded == MereRunCapabilityOption(flag: "--y", label: "Y", kind: .integer))
    #expect(decoded.defaultValue == nil && decoded.group == nil && decoded.tier == nil)
    #expect(decoded.range == nil && decoded.dependsOn == nil)
}

/// Capabilities whose artifact path is not caller-chosen, with the reason. Any
/// other file or directory output has to name the option that carries its
/// destination so a shell can request one.
private let selfLocatingOutputCapabilityIDs: [String: String] = [
    "adapter.pull": "Installs into the managed adapter store and prints the verified path.",
    "image.run-plan": "The saved plan names its own output path; --materialize adds a run directory."
]

@Test func capabilityFileOutputsDeclareADestinationFlag() {
    for command in MereRunCapabilityCatalog.document.commands {
        let output = command.output
        guard output.kind == .file || output.kind == .directory else { continue }
        if output.flag == nil {
            #expect(
                selfLocatingOutputCapabilityIDs[command.id] != nil,
                "\(command.id) writes a \(output.kind.rawValue) but names no destination flag"
            )
        }
        #expect(!output.optional, "\(command.id) declares a \(output.kind.rawValue) it does not always write")
    }

    for (id, reason) in selfLocatingOutputCapabilityIDs {
        let command = MereRunCapabilityCatalog.command(id: id)
        #expect(command != nil, "Exemption \(id) no longer matches a capability: \(reason)")
        #expect(command?.output.flag == nil, "\(id) names a destination flag now; remove its exemption.")
    }
}

@Test func capabilityOutputFlagsAreDeclaredOptions() {
    for command in MereRunCapabilityCatalog.document.commands {
        guard let flag = command.output.flag else {
            #expect(!command.output.optional, "\(command.id) declares an optional output with no flag to request it")
            continue
        }
        #expect(
            command.options.contains { $0.flag == flag },
            "\(command.id) writes its output to \(flag), which it does not declare as an option"
        )
    }
}

@Test func outputMetadataSerializesAdditivelyAndDecodesLegacyJSON() throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]

    // A mandatory artifact encodes exactly the keys the previous schema had, plus `flag`.
    let image = String(decoding: try encoder.encode(MereRunCapabilityCatalog.imageGenerate.output), as: UTF8.self)
    #expect(image == #"{"file_extension":"png","flag":"--output","kind":"file"}"#)

    // `optional` appears only where a run may write nothing.
    let diarize = String(decoding: try encoder.encode(MereRunCapabilityCatalog.speechDiarize.output), as: UTF8.self)
    #expect(diarize == #"{"flag":"--output","kind":"text","optional":true}"#)

    let chat = String(decoding: try encoder.encode(MereRunCapabilityCatalog.textChat.output), as: UTF8.self)
    #expect(chat == #"{"kind":"text"}"#)

    let legacy = Data(#"{"kind":"file","file_extension":"png"}"#.utf8)
    let decoded = try JSONDecoder().decode(MereRunCapabilityOutput.self, from: legacy)
    #expect(decoded == MereRunCapabilityOutput(kind: .file, fileExtension: "png"))
    #expect(decoded.flag == nil && !decoded.optional)
}

/// The commands that print their whole result to stdout and write nothing, even
/// though a neighbouring command in the same family writes a file.
@Test func stdoutOnlyCapabilitiesDeclareNoDestination() {
    for id in [
        "text.chat", "text.code", "vision.inspect", "music.analyze", "sfx.clap.score",
        "image.dataset.discover", "eval.pack.validate", "speech.listen",
        "model.benchmark.chat", "model.benchmark.code", "model.benchmark.fused",
        "model.benchmark.tool-calls", "model.benchmark.tool-continuations",
        "model.benchmark.api-workload"
    ] {
        let command = MereRunCapabilityCatalog.command(id: id)
        #expect(command?.output.kind == .text, "\(id) no longer declares a text output")
        #expect(command?.output.flag == nil, "\(id) names a destination flag now; declare the artifact it writes")
    }
}

@Test func speechTranscribeSampleRatePinsTheOnlyRateTheCLIAccepts() throws {
    // `SpeechTranscribe.validate()` requires `--sample-rate 16000` on raw stdin
    // and rejects the flag entirely off it.
    let sampleRate = try #require(
        MereRunCapabilityCatalog.speechTranscribe.options.first { $0.flag == "--sample-rate" }
    )
    #expect(sampleRate.range?.min == 16_000)
    #expect(sampleRate.range?.max == 16_000)
    #expect(sampleRate.dependsOn == "--stream")
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
    #expect(generate.options.first { $0.flag == "--h3-weight-mode" }?.choices == [
        "auto", "quantized", "resident-bf16"
    ])
    #expect(generate.options.first { $0.flag == "--reference" }?.repeatable == true)
}

@Test func postReleaseCapabilitiesRemainInTheSharedAppContract() {
    #expect(MereRunCapabilityCatalog.audioEnhance.command == ["audio", "enhance"])
    #expect(MereRunCapabilityCatalog.musicSeparate.command == ["music", "separate"])
    #expect(MereRunCapabilityCatalog.modelOptimize.command == ["model", "optimize"])
    #expect(MereRunCapabilityCatalog.textChat.options.contains { $0.flag == "--reasoning-effort" })
    #expect(MereRunCapabilityCatalog.textTrainLoRA.options.contains { $0.flag == "--reasoning-effort" })
    #expect(MereRunCapabilityCatalog.evaluationRun.command == ["eval", "run"])
    #expect(MereRunCapabilityCatalog.evaluationRun.options.contains { $0.flag == "--adapter" })
}
