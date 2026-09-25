import Foundation
import MereRunContract
import MereRunCore
import Testing

@testable import MereRunCLI

/// Routing the generated gate cases don't reach: the checks the audio, SFX, OCR, and TESSERA
/// commands no longer run themselves, selector-routed OCR, and agreement between each
/// command's own routing and the contract.

private func report(_ commandLine: String...) throws -> MereRunFamilyResolutionReport {
    try report(commandLine)
}

private func report(_ commandLine: [String]) throws -> MereRunFamilyResolutionReport {
    try #require(CLICapabilityGate.evaluate(commandLine: commandLine)).report
}

// MARK: - Checks the commands left to the gate

@Test func theGateRefusesWhatTheCommandsUsedToRefuseThemselves() throws {
    let cases: [(report: MereRunFamilyResolutionReport, violation: String)] = [
        (try report("audio", "enhance", "a.wav", "--model", "audio-enhance-universr-audio", "--overlap", "2"),
         "--overlap is not supported by UniverSR. It applies to AP-BWE."),
        (try report("audio", "enhance", "a.wav", "--input-rate", "8000"),
         "--input-rate 8000 is not supported by AP-BWE; it runs 16000. Remove --input-rate or pass 16000."),
        (try report("sfx", "generate", "door", "--negative-prompt", "music"),
         "--negative-prompt is not supported by Woosh DFlow. It applies to MMAudio."),
        (try report("sfx", "generate", "door", "-m", "sfx-mmaudio-large-44k-v2", "--renoise", "0.5"),
         "--renoise is not supported by MMAudio. It applies to Woosh DFlow."),
        (try report("sfx", "video", "generate", "steps", "v.mp4", "--negative-prompt", "music"),
         "--negative-prompt is not supported by Woosh DVFlow. It applies to MMAudio."),
        (try report("sfx", "video", "generate", "steps", "v.mp4", "--model", "sfx-mmaudio-large-44k-v2", "--renoise", "1"),
         "--renoise is not supported by MMAudio. It applies to Woosh DVFlow."),
        (try report("sfx", "video", "generate", "steps", "v.mp4", "--model", "sfx-woosh-flow"),
         "sfx-woosh-flow can't run sfx video generate: It generates sound from text; use `sfx generate`."),
        (try report("sfx", "generate", "door", "--model", "sfx-woosh-vflow-8s"),
         "sfx-woosh-vflow-8s can't run sfx generate: It generates sound from video; use `sfx video generate`."),
        (try report("geo", "tessera", "in.safetensors", "-o", "out.safetensors",
                    "--model", "vision-embed-tessera-v2-teacher", "--dimensions", "128"),
         "--dimensions 128 is not supported by TESSERA v2 Teacher; it runs 1024. Remove --dimensions or pass 1024.")
    ]
    for (report, violation) in cases {
        #expect(report.violations == [violation])
    }
    #expect(throws: CLICapabilityGate.Rejection.self) {
        try CLICapabilityGate.check(arguments: ["mere.run", "audio", "enhance", "a.wav", "--input-rate", "24000"])
    }
}

@Test func optionsAFamilyValidatesAndThenIgnoresOnlyWarn() throws {
    let flash = try report("audio", "edit", "Say hi", "--duration", "1", "--model", "audio-auk-flash", "--steps", "32")
    #expect(flash.violations.isEmpty)
    #expect(flash.warnings == ["--steps 32 has no effect with AuK Flash; it runs 4. Remove --steps or pass 4."])
    #expect(try report("audio", "edit", "Say hi", "--duration", "1", "--model", "audio-auk-flash", "--steps", "4").warnings.isEmpty)
    #expect(try report("audio", "edit", "Say hi", "--duration", "1", "--steps", "64").warnings.isEmpty)

    let flow = try report("sfx", "generate", "door", "--model", "sfx-woosh-flow", "--renoise", "0.5")
    #expect(flow.violations.isEmpty && flow.warnings == ["--renoise has no effect with Woosh Flow. It applies to Woosh DFlow."])
    let apBWE = try report("audio", "enhance", "a.wav", "--input-rate", "16000", "--seed", "7")
    #expect(apBWE.violations.isEmpty && apBWE.warnings == ["--seed has no effect with AP-BWE. It applies to UniverSR."])
}

// MARK: - OCR

@Test func ocrRoutesBySelectorsAndChecksEachFamilysModelFlag() throws {
    let cases: [([String], String?)] = [
        ([], "lighton"),
        (["--backend", "glm"], "glm"),
        (["-b", "infinity"], "infinity-native"),
        (["--backend", "infinity", "--infinity-runtime", "external"], "infinity-external"),
        (["--compare"], "compare-glm"),
        (["--compare", "--backend", "glm"], "compare-glm"),
        (["--compare", "--backend", "infinity"], "compare-infinity-native"),
        (["--compare", "--backend", "infinity", "--infinity-runtime=external"], "compare-infinity-external")
    ]
    for (arguments, family) in cases {
        let result = try report(["vision", "ocr", "page.png"] + arguments)
        #expect(result.family == family && result.violations.isEmpty && result.warnings.isEmpty, "\(arguments): \(result)")
    }

    #expect(try report("vision", "ocr", "page.png", "--model", "vision-ocr-infinity-pro-int8").violations == [
        "vision-ocr-infinity-pro-int8 runs on Infinity-Parser2 native, not LightOnOCR; change the model or the selector flags."
    ])
    #expect(try report("vision", "ocr", "page.png", "-b", "infinity", "--infinity-model", "vision-ocr-lighton").violations == [
        "vision-ocr-lighton runs on LightOnOCR, not Infinity-Parser2 native; change the model or the selector flags."
    ])
    let glm = try report("vision", "ocr", "page.png", "--backend", "glm", "--temperature", "0.5", "--infinity-batch-size", "2")
    #expect(glm.violations.isEmpty && glm.warnings == [
        "--infinity-batch-size has no effect with GLM-OCR. It applies to Infinity-Parser2 external and "
            + "LightOnOCR vs Infinity-Parser2 external.",
        "--temperature has no effect with GLM-OCR. It applies to LightOnOCR, Infinity-Parser2 native, "
            + "LightOnOCR vs GLM-OCR, LightOnOCR vs Infinity-Parser2 native and LightOnOCR vs Infinity-Parser2 external."
    ])
    let local = try report("vision", "ocr", "page.png", "--model", "/models/lighton")
    #expect(local.family == "lighton" && local.model == "/models/lighton" && local.source == .selector)
}

/// `VisionOCR.plan` is what `run()` executes; every backend, runtime, and comparison lands in the
/// family the contract resolves for the same command line.
@Test func ocrPlansAgreeWithTheContract() throws {
    for backend in OCRBackend.allCases {
        for runtime in InfinityParserRuntime.allCases {
            for compare in [false, true] {
                let arguments = ["page.png", "--backend", backend.rawValue, "--infinity-runtime", runtime.rawValue]
                    + (compare ? ["--compare"] : [])
                let plan = try VisionOCR.parse(arguments).plan
                let expected = switch plan {
                case .lightOn: "lighton"
                case .single(.infinity): "infinity-\(runtime.rawValue)"
                case .single(let other): other.rawValue
                case .comparison(.infinity): "compare-infinity-\(runtime.rawValue)"
                case .comparison(let other): "compare-\(other.rawValue)"
                }
                #expect(try report(["vision", "ocr"] + arguments).family == expected, "\(arguments) → \(plan)")
            }
        }
    }
}

// MARK: - Audio

/// Both audio commands route by exact id; the contract's families list exactly the ids each
/// command accepts, and AuK's variant is the family.
@Test func audioCommandsRouteTheIdsTheContractLists() throws {
    let enhance = try #require(MereRunCapabilityCatalog.audioEnhance.routing)
    #expect(Set(enhance.families.flatMap(\.models)) == Set(AudioEnhance.supportedModels))
    let edit = try #require(MereRunCapabilityCatalog.audioEdit.routing)
    #expect(Set(edit.families.flatMap(\.models)) == Set(AudioEdit.supportedModels))
    for family in edit.families {
        for model in family.models {
            let variant = try AudioEdit.parse(["Say hi", "--duration", "1", "--model", model]).options.variant
            #expect("auk-\(variant.rawValue)" == family.id, "\(model)")
        }
    }
}

@Test func tesseraLeavesItsMachineChosenDefaultToTheCommand() throws {
    let blank = try report("geo", "tessera", "in.safetensors", "-o", "out.safetensors", "--dimensions", "1024")
    #expect(blank.source == .unidentified && blank.violations.isEmpty)
    let student = try report("geo", "tessera", "in.safetensors", "-o", "out.safetensors", "--model", "vision-embed-tessera-v2-nano")
    #expect(student.family == "tessera-student" && student.violations.isEmpty)
}
