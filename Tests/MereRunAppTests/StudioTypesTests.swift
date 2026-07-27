@testable import MereRunApp
import MereRunContract
import XCTest

final class StudioTypesTests: XCTestCase {
    func testStudioModesMapToPublicTemplates() {
        XCTAssertEqual(StudioMode.createImage.defaultTemplateID, .imageGenerate)
        XCTAssertEqual(StudioMode.chat.defaultTemplateID, .textChat)
        XCTAssertEqual(StudioMode.code.defaultTemplateID, .textCode)
        XCTAssertEqual(StudioMode.speak.defaultTemplateID, .speechSynthesize)
        XCTAssertEqual(StudioMode.listen.defaultTemplateID, .speechTranscribe)
        XCTAssertEqual(StudioMode.readImage.defaultTemplateID, .visionInspect)
        XCTAssertEqual(StudioMode.findObjects.defaultTemplateID, .visionGround)
        XCTAssertEqual(StudioMode.segment.defaultTemplateID, .visionSegment)
        XCTAssertEqual(StudioMode.track.defaultTemplateID, .visionTrack)
        XCTAssertEqual(StudioMode.music.defaultTemplateID, .musicGenerate)
        XCTAssertEqual(StudioMode.video.defaultTemplateID, .videoGenerate)
    }

    func testCodeStudioDefaultsUsePublicModelID() throws {
        let codeTemplate = try XCTUnwrap(CommandCatalog.template(id: .textCode))
        XCTAssertEqual(codeTemplate.defaultModel, StudioCodeDefaults.fallbackModelID)

        var draft = StudioDraft()
        draft.reset(for: .code)
        XCTAssertEqual(draft.model, StudioCodeDefaults.fallbackModelID)
        XCTAssertEqual(StudioCommandAdapter.requiredModel(for: .code, draft: draft), StudioCodeDefaults.fallbackModelID)
    }

    func testChatStudioDefaultsToStreamingOutput() throws {
        let chatTemplate = try XCTUnwrap(CommandCatalog.template(id: .textChat))
        let draft = chatTemplate.defaultDraft()

        XCTAssertTrue(draft.stream)
        XCTAssertTrue(chatTemplate.arguments(from: draft).contains("--stream"))
    }

    func testChatBuildsVisionAndToolLoopFlags() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .textChat))
        var draft = template.defaultDraft()
        draft.imagePath = "/tmp/in.png"
        draft.tools = "write_file,shell_exec"
        draft.toolLoop = true
        draft.allowShellExec = true
        draft.sandboxDir = "/tmp/box"
        let args = template.arguments(from: draft)
        assertPair(args, "--image", "/tmp/in.png")
        assertPair(args, "--tools", "write_file,shell_exec")
        assertPair(args, "--sandbox-dir", "/tmp/box")
        XCTAssertTrue(args.contains("--tool-loop"))
        XCTAssertTrue(args.contains("--allow-shell-exec"))
    }

    func testChatOmitsVisionAndToolFlagsByDefault() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .textChat))
        let args = template.arguments(from: template.defaultDraft())
        XCTAssertFalse(args.contains("--image"))
        XCTAssertFalse(args.contains("--tools"))
        XCTAssertFalse(args.contains("--tool-loop"))
        XCTAssertFalse(args.contains("--allow-shell-exec"))
    }

    func testChatBuildsJSONLoRAKVPreflightAndToolPermissionFlags() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .textChat))
        var draft = template.defaultDraft()
        draft.responseFormat = .jsonObject
        draft.thinkingMode = .hide
        draft.contextSize = 262_144
        draft.topK = 20
        draft.kvBits = 4
        draft.kvQuantScheme = "polar"
        draft.kvGroupSize = 64
        draft.quantizedKVStart = 1_024
        draft.modelRoot = "/tmp/model"
        draft.loraPath = "adapter-assistant"
        draft.loraScale = 0.75
        draft.allowAbsoluteToolPaths = true
        draft.autoApproveTools = true
        draft.preflight = true
        draft.json = true
        draft.requireInstalled = true

        let args = template.arguments(from: draft)
        assertPair(args, "--response-format", "json_object")
        assertPair(args, "--context-size", "262144")
        assertPair(args, "--top-k", "20")
        assertPair(args, "--kv-bits", "4")
        assertPair(args, "--kv-quant-scheme", "polar")
        assertPair(args, "--kv-group-size", "64")
        assertPair(args, "--quantized-kv-start", "1024")
        assertPair(args, "--model-root", "/tmp/model")
        assertPair(args, "--lora", "adapter-assistant")
        assertPair(args, "--lora-scale", "0.75")
        XCTAssertTrue(args.contains("--no-thinking"))
        XCTAssertTrue(args.contains("--allow-absolute-tool-paths"))
        XCTAssertTrue(args.contains("--auto-approve-tools"))
        XCTAssertTrue(args.contains("--preflight"))
        XCTAssertTrue(args.contains("--json"))
        XCTAssertTrue(args.contains("--require-installed"))
        XCTAssertFalse(args.contains("--thinking"))
    }

    func testTextAppArgumentsAreDeclaredBySharedCapabilityContract() throws {
        let fixtures: [(CommandTemplateID, String, (inout CommandDraft) -> Void)] = [
            (.textChat, "text.chat", {
                $0.responseFormat = .jsonObject
                $0.thinkingMode = .show
                $0.loraPath = "/tmp/chat.safetensors"
                $0.preflight = true
                $0.json = true
            }),
            (.textCode, "text.code", { _ in }),
            (.textEmbed, "text.embed", { _ in }),
            (.textAnonymize, "text.anonymize", {
                $0.replacement = "<{label}:{index}>"
                $0.all = true
            }),
            (.textTrainLoRA, "text.train-lora", {
                $0.inputPath = "/tmp/train.jsonl"
                $0.evalPath = "/tmp/eval.jsonl"
                $0.dryRun = true
                $0.visualize = true
                $0.json = true
            })
        ]

        for (templateID, capabilityID, mutate) in fixtures {
            let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
            var draft = template.defaultDraft()
            mutate(&draft)
            XCTAssertNil(template.validationMessage(for: draft), "\(templateID) should validate")
            let arguments = template.arguments(from: draft)
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: capabilityID))
            XCTAssertEqual(Array(arguments.prefix(capability.command.count)), capability.command)
            let declared = Set(capability.options.map(\.flag))
            for flag in arguments where flag.hasPrefix("--") {
                XCTAssertTrue(declared.contains(flag), "\(templateID) emitted undeclared flag \(flag)")
            }
        }
    }

    func testNewAdvancedTemplatesBuildExpectedCommands() throws {
        func args(_ id: CommandTemplateID, _ mutate: (inout CommandDraft) -> Void = { _ in }) throws -> [String] {
            let template = try XCTUnwrap(CommandCatalog.template(id: id))
            var draft = template.defaultDraft()
            mutate(&draft)
            return template.arguments(from: draft)
        }
        XCTAssertEqual(try args(.musicAnalyze) { $0.inputPath = "/a.wav" }.prefix(3).map { $0 }, ["music", "analyze", "/a.wav"])
        XCTAssertEqual(try args(.musicTranscribe) { $0.inputPath = "/a.wav" }.prefix(3).map { $0 }, ["music", "transcribe", "/a.wav"])
        XCTAssertEqual(try args(.musicRealtime).prefix(2).map { $0 }, ["music", "realtime"])
        XCTAssertFalse(try args(.musicRealtime).contains("--no-play"))
        XCTAssertTrue(try args(.musicRealtime) { $0.musicPlay = false }.contains("--no-play"))
        XCTAssertEqual(
            try args(.musicTrainAdapter) {
                $0.inputPath = "/dataset.jsonl"
                $0.outputPath = "/adapter.safetensors"
            }.prefix(2).map { $0 },
            ["music", "train-adapter"]
        )
        XCTAssertEqual(try args(.musicServe).prefix(2).map { $0 }, ["music", "serve"])
        XCTAssertEqual(try args(.sfxAEEncode) { $0.inputPath = "/a.wav" }.prefix(3).map { $0 }, ["sfx", "ae", "encode"])
        XCTAssertEqual(try args(.sfxAEDecode) { $0.inputPath = "/a.npy" }.prefix(3).map { $0 }, ["sfx", "ae", "decode"])
        XCTAssertEqual(try args(.sfxClapScore) { $0.prompt = "door"; $0.inputPath = "/a.wav" }.prefix(3).map { $0 }, ["sfx", "clap", "score"])
        XCTAssertEqual(try args(.sfxConditionText) { $0.prompt = "door" }.prefix(3).map { $0 }, ["sfx", "condition", "text"])
        XCTAssertEqual(try args(.modelBenchmark).prefix(3).map { $0 }, ["model", "benchmark", "q36-mtp"])
        XCTAssertEqual(try args(.pluginList).prefix(2).map { $0 }, ["plugin", "list"])
        XCTAssertEqual(try args(.pluginInstall) { $0.prompt = "mere-runpod"; $0.force = true }, ["plugin", "install", "mere-runpod", "--yes"])
        XCTAssertEqual(try args(.pluginDoctor) { $0.prompt = "mere-runpod" }, ["plugin", "doctor", "mere-runpod"])
        XCTAssertEqual(try args(.openWebui).prefix(2).map { $0 }, ["open-webui", "quickstart"])
    }

    func testMusicAppArgumentsAreDeclaredBySharedCapabilityContract() throws {
        let fixtures: [(CommandTemplateID, String, (inout CommandDraft) -> Void)] = [
            (.musicGenerate, "music.generate", {
                $0.musicTask = "repaint"
                $0.musicSourceAudio = "/tmp/source.wav"
                $0.musicReferenceAudioPaths = "/tmp/voice.wav\n/tmp/timbre.wav"
                $0.musicLRCFile = "/tmp/lyrics.lrc"
                $0.musicLRCOutput = "/tmp/out.lrc"
                $0.musicAdapterPaths = "/tmp/a.safetensors\n/tmp/b.safetensors"
                $0.musicAdapterScales = "0.5\n0.75"
                $0.musicStems = "Drums,Bass,Vocals"
                $0.musicDAWBundle = "/tmp/session"
                $0.musicLMMode = "use"
                $0.musicAnalyzeSourceAudio = true
                $0.musicOverrideSteps = true
                $0.useDuration = true
                $0.musicCandidates = 3
                $0.musicKeepCandidates = true
                $0.musicRetakeSeed = "99"
                $0.musicFlowEdit = true
                $0.musicSourceCaption = "original demo"
                $0.musicSourceLyrics = "old words"
                $0.musicBPM = "122"
                $0.musicKey = "D minor"
                $0.musicTimeSignature = "4"
                $0.musicNoTiledVAE = true
                $0.musicNoRecipe = true
            }),
            (.musicAnalyze, "music.analyze", {
                $0.inputPath = "/tmp/song.wav"
                $0.useDuration = true
                $0.musicIncludeRawLM = true
                $0.musicIncludeAudioCodes = true
            }),
            (.musicTranscribe, "music.transcribe", {
                $0.inputPath = "/tmp/song.wav"
                $0.musicSampling = true
                $0.musicStrictEOS = true
                $0.musicContextOutput = "/tmp/context.json"
            }),
            (.musicRealtime, "music.realtime", {
                $0.musicPlay = false
                $0.musicInteractive = true
                $0.musicMIDIInput = "OP-1"
                $0.musicMIDICCMappings = "1=temp:0.2:1.4"
            }),
            (.musicTrainAdapter, "music.train-adapter", {
                $0.inputPath = "/tmp/dataset.jsonl"
                $0.outputPath = "/tmp/music.safetensors"
                $0.musicTrainingKind = "lokr"
            }),
            (.musicServe, "music.serve", {
                $0.musicAdapterPaths = "/tmp/music.safetensors"
                $0.musicAdapterScales = "0.8"
            })
        ]

        for (templateID, capabilityID, mutate) in fixtures {
            let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
            var draft = template.defaultDraft()
            mutate(&draft)
            let arguments = template.arguments(from: draft)
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: capabilityID))
            XCTAssertEqual(Array(arguments.prefix(capability.command.count)), capability.command)
            let declared = Set(capability.options.map(\.flag))
            for flag in arguments where flag.hasPrefix("--") {
                XCTAssertTrue(declared.contains(flag), "\(templateID) emitted undeclared flag \(flag)")
            }
        }
    }

    func testMusicGenerateBuildsCompleteProductionWorkflow() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "dream-pop vocals over live drums"
        draft.secondaryText = "[verse]\nwe rise"
        draft.musicQuality = "final"
        draft.musicTask = "repaint"
        draft.musicSourceAudio = "/tmp/demo.wav"
        draft.musicReferenceAudioPaths = "/tmp/vocal.wav\n/tmp/band.wav"
        draft.musicLMMode = "use"
        draft.musicAnalyzeSourceAudio = true
        draft.useDuration = true
        draft.durationSeconds = 42
        draft.musicOverrideSteps = true
        draft.steps = 50
        draft.musicCandidates = 4
        draft.musicKeepCandidates = true
        draft.musicAdapterPaths = "/tmp/style.safetensors\n/tmp/singer.safetensors"
        draft.musicAdapterScales = "0.6\n0.8"
        draft.musicStems = "Drums,Bass,Vocals"
        draft.musicDAWBundle = "/tmp/session"
        draft.musicFlowEdit = true
        draft.musicSourceCaption = "rough acoustic demo"
        draft.musicSourceLyrics = "old lyric"

        let args = template.arguments(from: draft)
        assertPair(args, "--quality", "final")
        assertPair(args, "--task-type", "repaint")
        assertPair(args, "--source-audio", "/tmp/demo.wav")
        XCTAssertEqual(args.filter { $0 == "--reference-audio" }.count, 2)
        XCTAssertEqual(args.filter { $0 == "--adapter" }.count, 2)
        XCTAssertEqual(args.filter { $0 == "--adapter-scale" }.count, 2)
        assertPair(args, "--duration", "42")
        assertPair(args, "--steps", "50")
        assertPair(args, "--candidates", "4")
        assertPair(args, "--stems", "Drums,Bass,Vocals")
        assertPair(args, "--daw-bundle", "/tmp/session")
        XCTAssertTrue(args.contains("--use-lm"))
        XCTAssertTrue(args.contains("--analyze-source-audio"))
        XCTAssertTrue(args.contains("--keep-candidates"))
        XCTAssertTrue(args.contains("--flow-edit"))
    }

    func testMusicStudioMapsProductionWorkflow() throws {
        var draft = StudioDraft()
        draft.reset(for: .music)
        draft.prompt = "cinematic art-pop single"
        draft.secondaryText = "[chorus]\nwe are alive"
        draft.musicQuality = "final"
        draft.musicTask = "cover"
        draft.musicSourceAudio = "/tmp/source.wav"
        draft.musicReferenceAudioPaths = "/tmp/timbre.wav"
        draft.musicLMMode = "use"
        draft.musicAnalyzeSourceAudio = true
        draft.useDuration = true
        draft.durationSeconds = 60
        draft.musicOverrideSteps = true
        draft.steps = 50
        draft.musicCandidates = 4
        draft.musicKeepCandidates = true
        draft.musicAdapterPaths = "/tmp/artist.safetensors"
        draft.musicAdapterScales = "0.7"
        draft.musicStems = "Drums,Bass,Vocals"
        draft.musicDAWBundle = "/tmp/daw"

        let request = try StudioCommandAdapter.makeRequest(mode: .music, draft: draft)
        let args = request.template.arguments(from: request.draft)
        assertPair(args, "--quality", "final")
        assertPair(args, "--task-type", "cover")
        assertPair(args, "--source-audio", "/tmp/source.wav")
        assertPair(args, "--duration", "60")
        assertPair(args, "--steps", "50")
        assertPair(args, "--candidates", "4")
        assertPair(args, "--adapter", "/tmp/artist.safetensors")
        assertPair(args, "--adapter-scale", "0.7")
        assertPair(args, "--stems", "Drums,Bass,Vocals")
        assertPair(args, "--daw-bundle", "/tmp/daw")
        XCTAssertTrue(args.contains("--use-lm"))
        XCTAssertTrue(args.contains("--analyze-source-audio"))
        XCTAssertTrue(args.contains("--keep-candidates"))
    }

    func testMusicEditingRequiresSourceAudioInStudioAndAdvanced() throws {
        var studio = StudioDraft()
        studio.reset(for: .music)
        studio.prompt = "new arrangement"
        studio.musicTask = "repaint"
        XCTAssertThrowsError(try StudioCommandAdapter.makeRequest(mode: .music, draft: studio))

        let template = try XCTUnwrap(CommandCatalog.template(id: .musicGenerate))
        var advanced = template.defaultDraft()
        advanced.musicTask = "cover"
        XCTAssertNotNil(template.validationMessage(for: advanced))
        advanced.musicSourceAudio = "/tmp/source.wav"
        XCTAssertNil(template.validationMessage(for: advanced))
    }

    func testMusicServerInjectsAPIKeyOutsideProcessArguments() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicServe))
        var draft = template.defaultDraft()
        draft.host = "0.0.0.0"
        draft.apiKey = "music-secret"

        let arguments = template.arguments(from: draft)
        XCTAssertFalse(arguments.contains("--api-key"))
        XCTAssertFalse(arguments.contains("music-secret"))
        XCTAssertEqual(
            CommandLaunchEnvironment.overrides(templateID: .musicServe, draft: draft),
            ["MERERUN_API_KEY": "music-secret"]
        )
        XCTAssertNil(template.validationMessage(for: draft))
    }

    func testVisionAndVFXArgumentsAreDeclaredBySharedCapabilityContract() throws {
        let fixtures: [(CommandTemplateID, String, (inout CommandDraft) -> Void)] = [
            (.visionInspect, "vision.inspect", { $0.inputPath = "/tmp/a.png" }),
            (.visionCaption, "vision.caption", {
                $0.inputPath = "/tmp/a.png"
                $0.visionAdditionalInputs = "/tmp/b.png"
                $0.visionPromptFile = "/tmp/prompt.txt"
                $0.visionFocus = "printed title\ncard border"
            }),
            (.visionOCR, "vision.ocr", {
                $0.inputPath = "/tmp/a.png"
                $0.backend = "infinity"
                $0.visionInfinityTask = "custom"
                $0.visionInfinityPrompt = "Return a table."
            }),
            (.visionGround, "vision.ground", { $0.inputPath = "/tmp/a.png" }),
            (.visionSegment, "vision.segment", {
                $0.inputPath = "/tmp/a.png"
                $0.visionBoxPrompts = "1,2,30,40,person"
                $0.visionPointPrompts = "10,20,positive,face"
                $0.visionMultimask = true
            }),
            (.visionTrack, "vision.track", {
                $0.inputPath = "/tmp/a.mp4"
                $0.preflight = true
                $0.json = true
            }),
            (.visionTrackLive, "vision.track-live", { _ in }),
            (.visionFaceDetect, "vision.face.detect", {
                $0.inputPath = "/tmp/a.png"
                $0.visionIncludeEmbeddings = true
            }),
            (.visionFaceEmbed, "vision.face.embed", {
                $0.inputPath = "/tmp/a.png"
                $0.visionFaceIndex = "1"
            }),
            (.visionFaceCompare, "vision.face.compare", {
                $0.inputPath = "/tmp/a.png"
                $0.visionSecondInputPath = "/tmp/b.png"
            }),
            (.visionFaceBatch, "vision.face.batch", {
                $0.inputPath = "/tmp/a.png"
                $0.visionAdditionalInputs = "/tmp/b.png"
                $0.visionIncludeEmbeddings = true
            }),
            (.visionPose, "vision.pose", {
                $0.inputPath = "/tmp/a.png"
                $0.visionPoseHands = false
            }),
            (.visionFlow, "vision.flow", {
                $0.inputPath = "/tmp/a.png"
                $0.visionSecondInputPath = "/tmp/b.png"
            }),
            (.visionDepthVideo, "vision.depth-video", { $0.inputPath = "/tmp/a.mp4" }),
            (.visionGeometry, "vision.geometry", { $0.inputPath = "/tmp/a.png" }),
            (.visionGeometryMultiview, "vision.geometry-multiview", {
                $0.inputPath = "/tmp/a.png"
                $0.visionAdditionalInputs = "/tmp/b.png"
            })
        ]

        for (templateID, capabilityID, mutate) in fixtures {
            let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
            var draft = template.defaultDraft()
            mutate(&draft)
            XCTAssertNil(template.validationMessage(for: draft), "\(templateID) should validate")
            let arguments = template.arguments(from: draft)
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: capabilityID))
            XCTAssertEqual(Array(arguments.prefix(capability.command.count)), capability.command)
            let declared = Set(capability.options.map(\.flag))
            for flag in arguments where flag.hasPrefix("--") {
                XCTAssertTrue(declared.contains(flag), "\(templateID) emitted undeclared flag \(flag)")
            }
        }
    }

    func testVisionGeometryAndPromptBuildersPreserveOrderedInputsAndCommaCoordinates() throws {
        let segment = try XCTUnwrap(CommandCatalog.template(id: .visionSegment))
        var segmentDraft = segment.defaultDraft()
        segmentDraft.inputPath = "/tmp/a.png"
        segmentDraft.prompt = ""
        segmentDraft.visionBoxPrompts = "1,2,30,40,person"
        segmentDraft.visionPointPrompts = "10,20,positive,face"
        let segmentArgs = segment.arguments(from: segmentDraft)
        assertPair(segmentArgs, "--box", "1,2,30,40,person")
        assertPair(segmentArgs, "--point", "10,20,positive,face")

        let geometry = try XCTUnwrap(CommandCatalog.template(id: .visionGeometryMultiview))
        var geometryDraft = geometry.defaultDraft()
        geometryDraft.inputPath = "/tmp/front.png"
        geometryDraft.visionAdditionalInputs = "/tmp/left.png\n/tmp/right.png"
        let geometryArgs = geometry.arguments(from: geometryDraft)
        XCTAssertEqual(
            Array(geometryArgs.prefix(5)),
            ["vision", "geometry-multiview", "/tmp/front.png", "/tmp/left.png", "/tmp/right.png"]
        )
    }

    func testVisionOCRDefaultsToARealCLIBackend() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionOCR))
        var draft = template.defaultDraft()
        draft.inputPath = "/tmp/page.png"
        XCTAssertEqual(draft.backend, "lighton")
        assertPair(template.arguments(from: draft), "--backend", "lighton")
    }

    func testOperationsArgumentsAreDeclaredBySharedCapabilityContract() throws {
        let fixtures: [(CommandTemplateID, String, (inout CommandDraft) -> Void)] = [
            (.adapterList, "adapter.list", { _ in }),
            (.adapterPull, "adapter.pull", {
                $0.prompt = "mere-platform-assistant"
                $0.force = true
            }),
            (.runList, "run.list", {
                $0.operationsExecutor = "relay:fleet"
                $0.operationsLimit = 25
            }),
            (.runInspect, "run.inspect", {
                $0.operationsReference = "relay://fleet/job-1"
            }),
            (.runWatch, "run.watch", {
                $0.operationsReference = "relay://fleet/job-1"
                $0.operationsJSONStream = true
            }),
            (.runFetch, "run.fetch", {
                $0.operationsReference = "relay://fleet/job-1"
                $0.operationsArtifacts = "preview\nfinal"
            }),
            (.runCancel, "run.cancel", {
                $0.operationsReference = "/tmp/runs/job-1"
            }),
            (.runRetry, "run.retry", {
                $0.operationsReference = "relay://fleet/job-1"
            }),
            (.worldServe, "world.serve", {
                $0.operationsPrepare = true
                $0.operationsStateDirectory = "/tmp/world"
            }),
            (.statusSnapshot, "status", { _ in }),
            (.qualityGate, "gate", {
                $0.operationsGateSuite = "text,vision"
                $0.operationsStrictPerformance = true
            }),
            (.modelStorage, "model.storage", { _ in }),
            (.modelGarbageCollect, "model.gc", { $0.force = true }),
            (.modelRuntimeGet, "model.runtime.get", { _ in }),
            (.modelRuntimeSet, "model.runtime.set", {
                $0.operationsRuntimeAlias = "assistant"
                $0.operationsPinned = true
                $0.operationsRuntimeTTL = "900"
                $0.operationsRuntimeContext = "32768"
                $0.operationsRuntimeMaxTokens = "2048"
                $0.operationsRuntimeTemperature = "0.4"
                $0.operationsRuntimeTopP = "0.9"
                $0.operationsRuntimeEngine = "text-chat-gemma4"
                $0.operationsRuntimeKVCacheMode = "auto"
            })
        ]

        for (templateID, capabilityID, mutate) in fixtures {
            let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
            var draft = template.defaultDraft()
            mutate(&draft)
            XCTAssertNil(template.validationMessage(for: draft), "\(templateID) should validate")
            let arguments = template.arguments(from: draft)
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: capabilityID))
            XCTAssertEqual(Array(arguments.prefix(capability.command.count)), capability.command)
            let declared = Set(capability.options.map(\.flag))
            for flag in arguments where flag.hasPrefix("--") {
                XCTAssertTrue(declared.contains(flag), "\(templateID) emitted undeclared flag \(flag)")
            }
        }
    }

    func testRunFetchRejectsConflictingArtifactSelections() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .runFetch))
        var draft = template.defaultDraft()
        draft.operationsReference = "relay://fleet/job-1"
        draft.operationsAllArtifacts = true
        draft.operationsArtifacts = "preview"

        XCTAssertNotNil(template.validationMessage(for: draft))
    }

    func testWorldAndStatusAPIKeysUseEnvironmentNotProcessArguments() throws {
        for id in [CommandTemplateID.worldServe, .statusSnapshot] {
            let template = try XCTUnwrap(CommandCatalog.template(id: id))
            var draft = template.defaultDraft()
            draft.host = "0.0.0.0"
            draft.apiKey = " operator-secret "

            XCTAssertFalse(template.arguments(from: draft).contains("operator-secret"))
            XCTAssertEqual(
                CommandLaunchEnvironment.overrides(templateID: id, draft: draft),
                ["MERERUN_API_KEY": "operator-secret"]
            )
            XCTAssertNil(template.validationMessage(for: draft))
        }
    }

    func testGraphAndFleetStayExternalProductBoundaries() throws {
        let graph = try XCTUnwrap(CommandCatalog.template(id: .graphStudio))
        let node = try XCTUnwrap(CommandCatalog.template(id: .nodeConsole))

        XCTAssertEqual(graph.externalURL?.absoluteString, "https://studio.mere.run/app")
        XCTAssertEqual(node.externalURL?.absoluteString, "https://relay.mere.run")
        XCTAssertTrue(graph.arguments(from: graph.defaultDraft()).isEmpty)
        XCTAssertTrue(node.arguments(from: node.defaultDraft()).isEmpty)
    }

    func testStudioServerStatusParsesSnapshot() {
        let json = """
        {"server":{"url":"http://127.0.0.1:8080","health":"ok","loadedModels":["text-chat-gemma4"]},\
        "knownModelCount":40,\
        "installedModels":[{"id":"a","category":"text","size":"1 GB"},{"id":"b","category":"image","size":"2 GB"}]}
        """
        let status = StudioServerStatus.parse(jsonStdout: "probing...\n" + json + "\n")
        XCTAssertEqual(status?.health, "ok")
        XCTAssertEqual(status?.loadedModels, ["text-chat-gemma4"])
        XCTAssertEqual(status?.installedCount, 2)
        XCTAssertEqual(status?.isReachable, true)
    }

    func testStudioServerStatusReturnsNilForNonJSON() {
        XCTAssertNil(StudioServerStatus.parse(jsonStdout: "connection refused"))
    }

    func testStudioVoiceProfileParsesTabSeparatedList() {
        let out = "11111111-1111-1111-1111-111111111111\tNarrator\t2026-06-01\n"
            + "22222222-2222-2222-2222-222222222222\tHero\t2026-06-02\n"
        let profiles = StudioVoiceProfile.parse(listOutput: out)
        XCTAssertEqual(profiles.count, 2)
        XCTAssertEqual(profiles.first?.name, "Narrator")
        XCTAssertEqual(profiles.last?.id, "22222222-2222-2222-2222-222222222222")
    }

    private func assertPair(_ args: [String], _ flag: String, _ value: String,
                            file: StaticString = #filePath, line: UInt = #line) {
        guard let index = args.firstIndex(of: flag), index + 1 < args.count else {
            XCTFail("flag \(flag) not found in \(args)", file: file, line: line)
            return
        }
        XCTAssertEqual(args[index + 1], value, file: file, line: line)
    }

    func testStudioRunRequestBuildsImageCommandWithDefaultOutput() throws {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "a porcelain teapot on linen"
        draft.secondaryText = "blurry"
        draft.width = 768
        draft.height = 512
        draft.steps = 8

        let request = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: draft)

        XCTAssertEqual(request.templateID, .imageGenerate)
        XCTAssertEqual(request.draft.model, "image-zimage-nano")
        XCTAssertEqual(request.draft.prompt, "a porcelain teapot on linen")
        XCTAssertEqual(request.draft.secondaryText, "blurry")
        XCTAssertEqual(request.draft.width, 768)
        XCTAssertEqual(request.draft.height, 512)
        XCTAssertEqual(request.draft.steps, 8)
        XCTAssertEqual(request.expectedOutputURL?.pathExtension, "png")
    }

    func testImageTrainLoRATemplateBuildsTrainingCommand() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageTrainLoRA))
        var draft = template.defaultDraft()
        draft.inputPath = "/tmp/krea-dataset"
        draft.outputPath = "/tmp/krea-style.safetensors"
        draft.width = 768
        draft.height = 768
        draft.steps = 1200
        draft.seed = "7"

        XCTAssertEqual(template.inputKind, .directory)
        XCTAssertEqual(template.defaultModel, "image-krea2-raw")
        XCTAssertEqual(
            template.arguments(from: draft),
            [
                "image", "train-lora",
                "--output", "/tmp/krea-style.safetensors",
                "--data", "/tmp/krea-dataset",
                "--width", "768",
                "--height", "768",
                "--training-steps", "1200",
                "--model", "image-krea2-raw",
                "--learning-rate", "0.0001",
                "--rank", "16",
                "--batch-size", "1",
                "--max-text-length", "512",
                "--scheduler-steps", "1000",
                "--seed", "7",
            ]
        )
    }

    func testImageTrainingRecipeIsNotSilentlyOverriddenByFormDefaults() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageTrainLoRA))
        var draft = template.defaultDraft()
        draft.inputPath = "/tmp/style"
        draft.trainingRecipe = "krea-fast-style"

        let recipeArgs = template.arguments(from: draft)
        assertPair(recipeArgs, "--recipe", "krea-fast-style")
        XCTAssertFalse(recipeArgs.contains("--model"))
        XCTAssertFalse(recipeArgs.contains("--width"))
        XCTAssertFalse(recipeArgs.contains("--height"))
        XCTAssertFalse(recipeArgs.contains("--training-steps"))
        XCTAssertFalse(recipeArgs.contains("--learning-rate"))
        XCTAssertFalse(recipeArgs.contains("--rank"))

        draft.overrideTrainingRecipe = true
        let overrideArgs = template.arguments(from: draft)
        XCTAssertTrue(overrideArgs.contains("--model"))
        XCTAssertTrue(overrideArgs.contains("--width"))
        XCTAssertTrue(overrideArgs.contains("--training-steps"))
        XCTAssertTrue(overrideArgs.contains("--learning-rate"))
        XCTAssertTrue(overrideArgs.contains("--rank"))
    }

    func testImageGenerateBuildsMultiReferenceStructuredPromptLoRAAndKreaControls() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.referenceImagePaths = "/tmp/face.png\n/tmp/style.png"
        draft.keepOriginalAspect = true
        draft.strength = 0.4
        draft.structuredPrompt = true
        draft.structuredPromptModel = "text-chat-q36-nano"
        draft.structuredPromptModelRoot = "/tmp/q36"
        draft.structuredPromptMaxTokens = 1024
        draft.structuredPromptOutputPath = "/tmp/caption.json"
        draft.loraPath = "adapter-portrait"
        draft.loraScale = 0.75
        draft.sigmaShift = 8
        draft.kreaConditioningMultiplier = 1.25
        draft.kreaConditioningLayerWeights = "1,1,2.5"
        draft.kreaBaseQuantizationBits = "8"
        draft.preflight = true
        draft.json = true
        draft.progressJSON = true

        let args = template.arguments(from: draft)
        XCTAssertEqual(args.filter { $0 == "--ref-image" }.count, 2)
        assertPair(args, "--structured-prompt-model", "text-chat-q36-nano")
        assertPair(args, "--structured-prompt-model-root", "/tmp/q36")
        assertPair(args, "--structured-prompt-max-tokens", "1024")
        assertPair(args, "--structured-prompt-output", "/tmp/caption.json")
        assertPair(args, "--lora", "adapter-portrait")
        assertPair(args, "--lora-scale", "0.75")
        assertPair(args, "--sigma-shift", "8")
        assertPair(args, "--krea-conditioning-multiplier", "1.25")
        assertPair(args, "--krea-conditioning-layer-weights", "1,1,2.5")
        assertPair(args, "--krea-base-quantization-bits", "8")
        XCTAssertTrue(args.contains("--keep-original-aspect"))
        XCTAssertTrue(args.contains("--structured-prompt"))
        XCTAssertTrue(args.contains("--preflight"))
        XCTAssertTrue(args.contains("--json"))
        XCTAssertTrue(args.contains("--progress-json"))
    }

    func testImageStudioMapsReferenceLoRAStructuredPromptAndPreflightControls() throws {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "editorial portrait"
        draft.referenceImagePaths = "/tmp/person.png\n/tmp/wardrobe.png"
        draft.keepOriginalAspect = true
        draft.structuredPrompt = true
        draft.structuredPromptModel = "text-chat-q36-nano"
        draft.structuredPromptMaxTokens = 1024
        draft.imageMaxSequenceLength = 768
        draft.loraPath = "adapter-editorial"
        draft.loraScale = 0.6
        draft.sigmaShift = 8
        draft.kreaConditioningMultiplier = 1.2
        draft.kreaConditioningLayerWeights = "1,1,2"
        draft.kreaBaseQuantizationBits = "4"
        draft.preflight = true
        draft.preflightJSON = true
        draft.progressJSON = true

        let request = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: draft)
        let args = request.template.arguments(from: request.draft)
        XCTAssertEqual(args.filter { $0 == "--ref-image" }.count, 2)
        assertPair(args, "--structured-prompt-model", "text-chat-q36-nano")
        assertPair(args, "--structured-prompt-max-tokens", "1024")
        assertPair(args, "--max-sequence-length", "768")
        assertPair(args, "--lora", "adapter-editorial")
        assertPair(args, "--lora-scale", "0.6")
        assertPair(args, "--sigma-shift", "8")
        assertPair(args, "--krea-conditioning-multiplier", "1.2")
        assertPair(args, "--krea-conditioning-layer-weights", "1,1,2")
        assertPair(args, "--krea-base-quantization-bits", "4")
        XCTAssertTrue(args.contains("--keep-original-aspect"))
        XCTAssertTrue(args.contains("--structured-prompt"))
        XCTAssertTrue(args.contains("--preflight"))
        XCTAssertTrue(args.contains("--json"))
        XCTAssertTrue(args.contains("--progress-json"))
    }

    func testImageAppArgumentsAreDeclaredBySharedCapabilityContract() throws {
        let fixtures: [(CommandTemplateID, String, (inout CommandDraft) -> Void)] = [
            (.imageGenerate, "image.generate", {
                $0.referenceImagePaths = "/tmp/a.png\n/tmp/b.png"
                $0.structuredPrompt = true
                $0.loraPath = "adapter-style"
                $0.preflight = true
                $0.json = true
            }),
            (.imageTrainLoRA, "image.train-lora", {
                $0.inputPath = "/tmp/dataset"
                $0.trainingRecipe = "klein-fast-style"
                $0.sampleInterval = 100
                $0.visualize = true
                $0.preflight = true
                $0.json = true
            }),
            (.imageValidate, "image.validate", {
                $0.all = true
                $0.referenceDirectoryPath = "/tmp/reference"
            }),
            (.imageDatasetDiscover, "image.dataset.discover", {
                $0.inputPath = "/tmp/datasets"
                $0.trainingOutputRoot = "/tmp/runs"
            }),
            (.imageRunPlan, "image.run-plan", {
                $0.inputPath = "/tmp/plan.json"
                $0.preflight = true
            }),
            (.imageVisualizeRun, "image.visualize-run", {
                $0.inputPath = "/tmp/run"
            }),
            (.imageReconstruct3D, "image.reconstruct-3d", {
                $0.inputPath = "/tmp/object.png"
                $0.dryRun = true
                $0.json = true
            }),
            (.imageReconstruct3DTrellis2, "image.reconstruct-3d-trellis2", {
                $0.inputPath = "/tmp/object.png"
                $0.dryRun = true
                $0.json = true
            }),
            (.imageReconstruct3DMultiview, "image.reconstruct-3d-multiview", {
                $0.referenceImagePaths = [
                    "/tmp/front.png", "/tmp/right.png", "/tmp/back.png", "/tmp/left.png"
                ].joined(separator: "\n")
                $0.dryRun = true
                $0.json = true
            })
        ]

        for (templateID, capabilityID, mutate) in fixtures {
            let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
            var draft = template.defaultDraft()
            mutate(&draft)
            XCTAssertNil(template.validationMessage(for: draft), "\(templateID) should validate")
            let arguments = template.arguments(from: draft)
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: capabilityID))
            XCTAssertEqual(Array(arguments.prefix(capability.command.count)), capability.command)
            let declared = Set(capability.options.map(\.flag))
            for flag in arguments where flag.hasPrefix("--") {
                XCTAssertTrue(declared.contains(flag), "\(templateID) emitted undeclared flag \(flag)")
            }
        }
    }

    func testReadImageVariantsResolveToDistinctTemplates() throws {
        var draft = StudioDraft()
        draft.reset(for: .readImage)
        draft.inputPath = "/tmp/image.png"
        draft.prompt = "What text is visible?"

        draft.readImageAction = .inspect
        XCTAssertEqual(try StudioCommandAdapter.makeRequest(mode: .readImage, draft: draft).templateID, .visionInspect)

        draft.readImageAction = .ocr
        XCTAssertEqual(try StudioCommandAdapter.makeRequest(mode: .readImage, draft: draft).templateID, .visionOCR)

        draft.readImageAction = .caption
        XCTAssertEqual(try StudioCommandAdapter.makeRequest(mode: .readImage, draft: draft).templateID, .visionCaption)
    }

    func testMissingInputValidationIsFriendly() {
        var draft = StudioDraft()
        draft.reset(for: .listen)

        XCTAssertThrowsError(try StudioCommandAdapter.makeRequest(mode: .listen, draft: draft)) { error in
            XCTAssertEqual(error as? StudioCommandError, .missingInput("audio"))
        }
    }

    func testListenStudioUsesParakeetForCapabilityAndPullChecks() throws {
        var draft = StudioDraft()
        draft.reset(for: .listen)

        XCTAssertEqual(
            StudioCommandAdapter.capabilityRequirement(for: .listen, draft: draft),
            .managedModel("speech-asr-parakeet")
        )

        let request = try XCTUnwrap(StudioCommandAdapter.pullRequest(for: .listen, draft: draft))
        XCTAssertEqual(request.draft.model, "speech-asr-parakeet")
    }

    func testReadImageInspectAndCaptionAreNotGated() {
        var draft = StudioDraft()
        draft.reset(for: .readImage)

        // Inspect/caption use a vision-language model the CLI auto-downloads on demand, so
        // they are not gated by the managed capability catalog (the run proceeds and the
        // CLI fetches the model itself).
        draft.readImageAction = .inspect
        XCTAssertNil(StudioCommandAdapter.capabilityRequirement(for: .readImage, draft: draft))

        draft.readImageAction = .caption
        XCTAssertNil(StudioCommandAdapter.capabilityRequirement(for: .readImage, draft: draft))

        // OCR has a managed default model, so it remains readiness-gated.
        draft.readImageAction = .ocr
        XCTAssertEqual(
            StudioCommandAdapter.capabilityRequirement(for: .readImage, draft: draft),
            .managedModel("vision-ocr-lighton")
        )
    }

    func testReadImageModelOverrideGatesOnThatManagedModel() throws {
        var draft = StudioDraft()
        draft.reset(for: .readImage)
        draft.readImageAction = .inspect
        draft.model = "vision-ocr-lighton"

        // Naming an explicit managed model gates the run on that model and makes it pullable.
        XCTAssertEqual(
            StudioCommandAdapter.capabilityRequirement(for: .readImage, draft: draft),
            .managedModel("vision-ocr-lighton")
        )
        let request = try XCTUnwrap(StudioCommandAdapter.pullRequest(for: .readImage, draft: draft))
        XCTAssertEqual(request.draft.model, "vision-ocr-lighton")
    }

    func testUnknownReadinessBlocksRuns() {
        XCTAssertTrue(ModelReadinessState.unknown("Could not check models.").blocksRun)
    }

    func testStudioProgressParserParsesPercentBytesAndSpeed() {
        let bytes = StudioProgressParser.parse("[image-zimage-nano] 45%  1.2 GB / 3.4 GB")
        XCTAssertEqual(bytes?.label, "image-zimage-nano")
        XCTAssertEqual(bytes?.fractionCompleted ?? -1, 0.45, accuracy: 0.001)
        XCTAssertEqual(bytes?.detail, "1.2 GB / 3.4 GB")

        let speed = StudioProgressParser.parse("[m] 10%  (45 MB/s)")
        XCTAssertEqual(speed?.fractionCompleted ?? -1, 0.10, accuracy: 0.001)

        let fractional = StudioProgressParser.parse(
            "[image-zimage-nano] 0.4%  4 MB / 1 GB  2 MB/s  ETA 8m 18s"
        )
        XCTAssertEqual(fractional?.fractionCompleted ?? -1, 0.004, accuracy: 0.0001)
        XCTAssertEqual(fractional?.detail, "4 MB / 1 GB 2 MB/s ETA 8m 18s")

        let generation = StudioProgressParser.parse(
            #"{"event":"progress","stage":"denoising","step":2,"total_steps":4}"#
        )
        XCTAssertEqual(generation?.label, "Generating")
        XCTAssertEqual(generation?.fractionCompleted ?? -1, 0.75, accuracy: 0.001)
        XCTAssertEqual(generation?.detail, "Step 3 of 4")

        // Non-progress lines are not misclassified.
        XCTAssertNil(StudioProgressParser.parse("just a normal log line"))
        XCTAssertNil(StudioProgressParser.parse("[info] starting up"))
    }

    func testStudioResultParserExtractsBarePathFromStdout() {
        // Media commands print only the artifact path on stdout (diagnostics go to stderr).
        let stdout = "/Users/me/Documents/render.png\n"
        XCTAssertEqual(StudioResultParser.outputPaths(fromStdout: stdout), ["/Users/me/Documents/render.png"])
    }

    func testStudioResultParserReturnsMostRecentPathFirst() {
        let stdout = "/tmp/first.wav\n/tmp/second.wav\n"
        XCTAssertEqual(
            StudioResultParser.outputPaths(fromStdout: stdout),
            ["/tmp/second.wav", "/tmp/first.wav"]
        )
    }

    func testStudioResultParserExtractsRightSideOfArrowPair() {
        // vision ocr/caption print `input -> output`; the artifact is the right-hand side.
        let stdout = "/in/page.png -> /out/page.txt\n"
        XCTAssertEqual(StudioResultParser.outputPaths(fromStdout: stdout), ["/out/page.txt"])
        let unicodeArrow = "/in/page.png → /out/page.md\n"
        XCTAssertEqual(StudioResultParser.outputPaths(fromStdout: unicodeArrow), ["/out/page.md"])
    }

    func testStudioResultParserIgnoresNonPathLines() {
        // Transcription emits prose on stdout, not a path — nothing should be parsed.
        let stdout = "This is the transcript of the audio file.\nSecond sentence here.\n"
        XCTAssertTrue(StudioResultParser.outputPaths(fromStdout: stdout).isEmpty)
    }

    func testChatConversationRequestStreamsAndCarriesConversationID() throws {
        var draft = StudioDraft()
        draft.reset(for: .chat)
        draft.prompt = "User: hi\n\nAssistant: hey\n\nUser: more"
        draft.secondaryText = "Be terse."
        let conversationID = UUID()
        let request = try StudioCommandAdapter.makeRequest(
            mode: .chat, draft: draft, conversationID: conversationID
        )
        XCTAssertEqual(request.conversationID, conversationID)
        XCTAssertTrue(request.draft.stream)
        XCTAssertEqual(request.draft.prompt, "User: hi\n\nAssistant: hey\n\nUser: more")
        XCTAssertEqual(request.draft.secondaryText, "Be terse.")
    }

    func testChatSingleShotRequestHasNoConversationID() throws {
        var draft = StudioDraft()
        draft.reset(for: .chat)
        draft.prompt = "hello"
        let request = try StudioCommandAdapter.makeRequest(mode: .chat, draft: draft)
        XCTAssertNil(request.conversationID)
    }

    func testChatConversationRequestAttachesImage() throws {
        var draft = StudioDraft()
        draft.reset(for: .chat)
        draft.prompt = "what is this?"
        draft.inputPath = "/tmp/pic.png"
        let request = try StudioCommandAdapter.makeRequest(mode: .chat, draft: draft, conversationID: UUID())
        XCTAssertEqual(request.draft.imagePath, "/tmp/pic.png")
        assertPair(request.template.arguments(from: request.draft), "--image", "/tmp/pic.png")
    }

    func testCodeConversationRequestIgnoresImage() throws {
        var draft = StudioDraft()
        draft.reset(for: .code)
        draft.prompt = "hi"
        draft.inputPath = "/tmp/pic.png"
        let request = try StudioCommandAdapter.makeRequest(mode: .code, draft: draft, conversationID: UUID())
        XCTAssertTrue(request.draft.imagePath.isEmpty)
    }

    func testSpeakCloneRequestMapsProfileAndReferenceAudio() throws {
        var draft = StudioDraft()
        draft.reset(for: .speak)
        draft.prompt = "hello world"
        draft.voiceMode = "clone"
        draft.voiceProfile = "narrator-id"
        draft.refAudioPath = "/tmp/ref.wav"
        draft.saveProfileName = "Narrator"
        let request = try StudioCommandAdapter.makeRequest(mode: .speak, draft: draft)
        let args = request.template.arguments(from: request.draft)
        assertPair(args, "--mode", "clone")
        assertPair(args, "--profile", "narrator-id")
        assertPair(args, "--ref-audio", "/tmp/ref.wav")
        assertPair(args, "--save-profile", "Narrator")
    }

    func testSpeakStyleRequestOmitsCloneFlags() throws {
        var draft = StudioDraft()
        draft.reset(for: .speak)
        draft.prompt = "hello world"
        let request = try StudioCommandAdapter.makeRequest(mode: .speak, draft: draft)
        let args = request.template.arguments(from: request.draft)
        XCTAssertFalse(args.contains("--profile"))
        XCTAssertFalse(args.contains("--ref-audio"))
    }

    func testSpeakCloneWithoutSourceFailsValidation() {
        var draft = StudioDraft()
        draft.reset(for: .speak)
        draft.prompt = "hello world"
        draft.voiceMode = "clone"
        XCTAssertThrowsError(try StudioCommandAdapter.makeRequest(mode: .speak, draft: draft))
    }

    func testSpeakCloneWithProfilePassesValidation() throws {
        var draft = StudioDraft()
        draft.reset(for: .speak)
        draft.prompt = "hello world"
        draft.voiceMode = "clone"
        draft.voiceProfile = "narrator-id"
        XCTAssertNoThrow(try StudioCommandAdapter.makeRequest(mode: .speak, draft: draft))
    }

    func testChatSchemaExposesTemperatureAndMaxTokens() throws {
        var draft = StudioDraft()
        draft.reset(for: .chat)
        draft.prompt = "hi"
        draft.temperature = 0.3
        draft.maxTokens = 1234
        let request = try StudioCommandAdapter.makeRequest(mode: .chat, draft: draft)
        let args = request.template.arguments(from: request.draft)
        assertPair(args, "--temperature", "0.3")
        assertPair(args, "--max-tokens", "1234")
    }

    func testChatStudioMapsJSONLoRAKVAndPermissionControls() throws {
        var draft = StudioDraft()
        draft.reset(for: .chat)
        draft.prompt = "Return one JSON object."
        draft.responseFormat = .jsonObject
        draft.thinkingMode = .hide
        draft.contextSize = 32_768
        draft.topK = 20
        draft.kvBits = 8
        draft.kvQuantScheme = "turboquant"
        draft.kvGroupSize = 64
        draft.quantizedKVStart = 512
        draft.loraPath = "adapter-assistant"
        draft.loraScale = 0.5
        draft.tools = "write_file,shell_exec"
        draft.toolLoop = true
        draft.allowShellExec = true
        draft.allowAbsoluteToolPaths = true
        draft.autoApproveTools = true
        draft.sandboxDir = "/tmp/tools"
        draft.stats = true
        draft.preflight = true
        draft.preflightJSON = true
        draft.requireInstalled = true

        let request = try StudioCommandAdapter.makeRequest(mode: .chat, draft: draft)
        let args = request.template.arguments(from: request.draft)

        assertPair(args, "--response-format", "json_object")
        assertPair(args, "--context-size", "32768")
        assertPair(args, "--top-k", "20")
        assertPair(args, "--kv-bits", "8")
        assertPair(args, "--kv-quant-scheme", "turboquant")
        assertPair(args, "--lora", "adapter-assistant")
        assertPair(args, "--lora-scale", "0.5")
        assertPair(args, "--sandbox-dir", "/tmp/tools")
        XCTAssertTrue(args.contains("--no-thinking"))
        XCTAssertTrue(args.contains("--tool-loop"))
        XCTAssertTrue(args.contains("--allow-shell-exec"))
        XCTAssertTrue(args.contains("--allow-absolute-tool-paths"))
        XCTAssertTrue(args.contains("--auto-approve-tools"))
        XCTAssertTrue(args.contains("--stats"))
        XCTAssertTrue(args.contains("--preflight"))
        XCTAssertTrue(args.contains("--json"))
        XCTAssertTrue(args.contains("--require-installed"))
    }

    func testImageSchemaExposesCfgAndStrength() throws {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "a cat"
        draft.inputPath = "/tmp/base.png"
        draft.cfgScale = 6.5
        draft.strength = 0.4
        let request = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: draft)
        let args = request.template.arguments(from: request.draft)
        assertPair(args, "--cfg", "6.5")
        assertPair(args, "--strength", "0.4")
    }

    func testListenSchemaExposesLanguageBackendTimestamps() throws {
        var draft = StudioDraft()
        draft.reset(for: .listen)
        draft.inputPath = "/tmp/a.wav"
        draft.language = "es"
        draft.backend = "mlx"
        draft.timestamps = false
        let request = try StudioCommandAdapter.makeRequest(mode: .listen, draft: draft)
        let args = request.template.arguments(from: request.draft)
        assertPair(args, "--language", "es")
        assertPair(args, "--backend", "mlx")
        XCTAssertTrue(args.contains("--no-timestamps"))
    }

    func testVideoSchemaUsesTypedProductSelectionAndNeverLegacyVariant() throws {
        var draft = StudioDraft()
        draft.reset(for: .video)
        draft.prompt = "a wave"
        draft.videoQuality = .final
        draft.videoOutputMode = .videoOnly
        draft.fps = 30
        draft.numFrames = 120
        let request = try StudioCommandAdapter.makeRequest(mode: .video, draft: draft)
        let args = request.template.arguments(from: request.draft)
        assertPair(args, "--quality", "final")
        assertPair(args, "--output-mode", "video-only")
        assertPair(args, "--fps", "30")
        assertPair(args, "--num-frames", "120")
        XCTAssertFalse(args.contains("--variant"))
    }

    func testVideoStudioBuildsNativeAudioAndEndKeyframeConditioning() throws {
        var draft = StudioDraft()
        draft.reset(for: .video)
        draft.prompt = "a kinetic live performance"
        draft.inputPath = "/tmp/start.png"
        draft.endImagePath = "/tmp/end.png"
        draft.endImageStrength = 0.8
        draft.audioPath = "/tmp/song.wav"
        draft.audioStartTime = 30
        draft.useDuration = true
        draft.durationSeconds = 5

        let request = try StudioCommandAdapter.makeRequest(mode: .video, draft: draft)
        let args = request.template.arguments(from: request.draft)

        assertPair(args, "--quality", "final")
        assertPair(args, "--output-mode", "audio-video")
        assertPair(args, "--audio", "/tmp/song.wav")
        assertPair(args, "--audio-start-time", "30")
        assertPair(args, "--image", "/tmp/start.png")
        assertPair(args, "--end-image", "/tmp/end.png")
        assertPair(args, "--end-image-strength", "0.8")
        assertPair(args, "--duration", "5")
        XCTAssertFalse(args.contains("--num-frames"))
    }

    func testVideoWanRequestOmitsLTXProductSelectors() throws {
        var draft = StudioDraft()
        draft.reset(for: .video)
        draft.prompt = "the camera walks forward"
        draft.model = "video-wan22-ti2v-5b-mlx"
        draft.inputPath = "/tmp/frame.png"
        draft.steps = 40
        draft.cfgScale = 5
        draft.scheduleShift = 5

        let request = try StudioCommandAdapter.makeRequest(mode: .video, draft: draft)
        let args = request.template.arguments(from: request.draft)

        XCTAssertFalse(args.contains("--quality"))
        XCTAssertFalse(args.contains("--output-mode"))
        assertPair(args, "--steps", "40")
        assertPair(args, "--guidance-scale", "5")
        assertPair(args, "--shift", "5")
    }

    func testVideoAppArgumentsAreDeclaredBySharedCapabilityContract() throws {
        var draft = StudioDraft()
        draft.reset(for: .video)
        draft.prompt = "a performance"
        draft.inputPath = "/tmp/start.png"
        draft.endImagePath = "/tmp/end.png"
        draft.audioPath = "/tmp/song.wav"
        draft.preflight = true
        draft.timings = true
        draft.timingsOutputPath = "/tmp/timings.json"

        let request = try StudioCommandAdapter.makeRequest(mode: .video, draft: draft)
        let args = request.template.arguments(from: request.draft)
        let declared = Set(MereRunCapabilityCatalog.videoGenerate.options.map(\.flag))

        for flag in args where flag.hasPrefix("--") {
            XCTAssertTrue(declared.contains(flag), "App emitted undeclared Video flag \(flag)")
        }
    }

    func testGuidedAdvancedVideoWorkflowsBuildContractDeclaredCommands() throws {
        let fixtures: [(CommandTemplateID, String, (inout CommandDraft) -> Void)] = [
            (.videoAnimate, "video.animate", {
                $0.prompt = "a dancer"
                $0.inputPath = "/tmp/reference.png"
                $0.referenceMaskPath = "/tmp/reference-mask.png"
                $0.drivingVideoPath = "/tmp/driving.mp4"
                $0.drivingMaskPath = "/tmp/driving-mask.mp4"
            }),
            (.videoCosmos3, "video.cosmos3", {
                $0.prompt = "a rover"
                $0.cosmosMode = "image-to-video"
                $0.cosmosImagePath = "/tmp/rover.png"
            }),
            (.videoPrepareMasks, "video.prepare-masks", {
                $0.inputPath = "/tmp/plan.json"
            }),
            (.videoExportLatents, "video.export-latents", {
                $0.prompt = "a landscape"
            }),
            (.videoSession, "video.session", { _ in })
        ]

        for (templateID, capabilityID, mutate) in fixtures {
            let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
            var draft = template.defaultDraft()
            mutate(&draft)
            XCTAssertNil(template.validationMessage(for: draft), "\(templateID) should validate")
            let arguments = template.arguments(from: draft)
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: capabilityID))
            XCTAssertEqual(Array(arguments.prefix(capability.command.count)), capability.command)
            let declared = Set(capability.options.map(\.flag))
            for flag in arguments where flag.hasPrefix("--") {
                XCTAssertTrue(
                    declared.contains(flag),
                    "\(templateID) emitted undeclared flag \(flag)"
                )
            }
        }
    }

    func testSchemaDefaultsMatchTemplateDraftSoSurfacesDoNotDrift() {
        var draft = StudioDraft()
        draft.reset(for: .chat)
        let base = CommandCatalog.template(id: StudioMode.chat.defaultTemplateID)?.defaultDraft()
        XCTAssertEqual(draft.temperature, base?.temperature)
        XCTAssertEqual(draft.topP, base?.topP)
        XCTAssertEqual(draft.maxTokens, base?.maxTokens)
        XCTAssertEqual(draft.responseFormat, base?.responseFormat)
        XCTAssertEqual(draft.thinkingMode, base?.thinkingMode)
    }

    func testLegacyLibraryItemDecodesWithoutConversationFields() throws {
        let legacy = StudioLibraryItem(
            id: UUID(), mode: .chat, prompt: "hello there",
            inputURL: nil, outputURL: nil, createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "mere.run text chat",
            outputText: "hi"
        )
        let data = try JSONEncoder().encode(legacy)
        // Nil optionals are omitted, so the encoded form matches a pre-conversation row.
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertFalse(json.contains("messages"))
        XCTAssertFalse(json.contains("systemPrompt"))

        let decoded = try JSONDecoder().decode(StudioLibraryItem.self, from: data)
        XCTAssertNil(decoded.messages)
        XCTAssertFalse(decoded.isConversation)
        XCTAssertEqual(decoded.displayTitle, "hello there")
    }

    func testConversationItemRoundTripsAndTitlesFromFirstUserMessage() throws {
        let messages = [
            StudioMessage(role: .user, content: "what is swift?"),
            StudioMessage(role: .assistant, content: "A language."),
        ]
        var item = StudioLibraryItem(
            id: UUID(), mode: .chat, prompt: "",
            inputURL: nil, outputURL: nil, createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "mere.run text chat",
            outputText: nil
        )
        item.messages = messages
        item.systemPrompt = "You are helpful."
        item.model = "text-chat-gemma4"

        let data = try JSONEncoder().encode(item)
        let decoded = try JSONDecoder().decode(StudioLibraryItem.self, from: data)
        XCTAssertEqual(decoded.messages, messages)
        XCTAssertEqual(decoded.systemPrompt, "You are helpful.")
        XCTAssertEqual(decoded.model, "text-chat-gemma4")
        XCTAssertTrue(decoded.isConversation)
        XCTAssertEqual(decoded.displayTitle, "what is swift?")
    }

    func testSfxStudioModeBuildsGenerateCommand() throws {
        var draft = StudioDraft()
        draft.reset(for: .sfx)
        draft.prompt = "thunder clap"

        let request = try StudioCommandAdapter.makeRequest(mode: .sfx, draft: draft)
        XCTAssertEqual(request.templateID, .sfxGenerate)
        let args = request.template.arguments(from: request.draft)
        XCTAssertEqual(Array(args.prefix(2)), ["sfx", "generate"])
        XCTAssertTrue(args.contains("thunder clap"))
        XCTAssertTrue(args.contains("--duration"))
    }

    func testSpeechCloneFlagsBuildWhenCloneMode() {
        guard let template = CommandCatalog.template(id: .speechSynthesize) else {
            return XCTFail("missing speechSynthesize template")
        }
        var draft = template.defaultDraft()
        draft.prompt = "hello"
        draft.voiceMode = "clone"
        draft.refAudioPath = "/tmp/voice.wav"
        draft.saveProfileName = "narrator"

        let args = template.arguments(from: draft)
        XCTAssertTrue(args.contains("--mode"))
        XCTAssertTrue(args.contains("clone"))
        XCTAssertTrue(args.contains("--ref-audio"))
        XCTAssertTrue(args.contains("/tmp/voice.wav"))
        XCTAssertTrue(args.contains("--save-profile"))
    }

    func testStudioModelInventoryParserFindsDownloadedRows() {
        let output = """
        ID                         Category        Status     Size
        ----------------------------------------------------------
        image-zimage-nano          image           installed  4.2 GB
        text-chat-gemma4           text-chat       missing    —
        vision-segment-sam31       vision-segment  installed  950 MB
        Usage restriction: vision-face-buffalo-l - non-commercial research use only
        """

        let rows = StudioModelInventoryParser.rows(from: output)

        XCTAssertEqual(rows.count, 3)
        XCTAssertEqual(rows[0].id, "image-zimage-nano")
        XCTAssertEqual(rows[0].status, "installed")
        XCTAssertEqual(rows[0].size, "4.2 GB")
        XCTAssertTrue(rows[2].isInstalled)
        XCTAssertFalse(rows[1].isInstalled)
    }

    func testModelPullTemplatePropagatesUsageTermsAcceptance() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .modelPull))
        var draft = template.defaultDraft()
        draft.model = "vision-face-buffalo-l"
        draft.acceptModelLicense = true

        XCTAssertTrue(template.arguments(from: draft).contains("--accept-model-license"))
    }

    func testAgentOnboardTemplatePropagatesUsageTermsAcceptance() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .agentOnboard))
        var draft = template.defaultDraft()
        draft.force = true
        draft.acceptModelLicense = true

        let arguments = template.arguments(from: draft)
        XCTAssertTrue(arguments.contains("--pull-recommended"))
        XCTAssertTrue(arguments.contains("--accept-model-license"))
    }

    func testStudioModelInventoryMarksRestrictedModels() {
        let output = """
        vision-face-buffalo-l vision-face missing 287 MB
        Usage terms: vision-face-buffalo-l - InsightFace weights are non-commercial. [model: research terms https://github.com/deepinsight/insightface#license]
        """
        let row = StudioModelInventoryParser.rows(from: output).first

        XCTAssertNotNil(row?.usageTerms)
        XCTAssertTrue(row?.usageTerms?.summary.contains("deepinsight/insightface#license") == true)
    }

    func testStudioModelInventoryParserFindsModelRoot() throws {
        let output = """
        Model Root: /Users/example/Library/Application Support/MereRun/models/image-zimage-nano
        Model ID: image-zimage-nano
        """

        let root = try XCTUnwrap(StudioModelInventoryParser.modelRoot(from: output))

        XCTAssertEqual(root.path, "/Users/example/Library/Application Support/MereRun/models/image-zimage-nano")
    }

    func testStudioRuntimeSettingsDecodesCLIJSON() throws {
        let data = """
        {
          "alias": "chat-default",
          "pinned": true,
          "ttlSeconds": 600,
          "maxContextTokens": 8192,
          "maxTokens": 512,
          "temperature": 0.4,
          "topP": 0.8,
          "engineOverride": "text-chat-gemma4"
        }
        """.data(using: .utf8)!

        let settings = try JSONDecoder().decode(StudioRuntimeSettings.self, from: data)

        XCTAssertEqual(settings.alias, "chat-default")
        XCTAssertTrue(settings.pinned)
        XCTAssertEqual(settings.ttlSeconds, 600)
        XCTAssertEqual(settings.maxContextTokens, 8192)
        XCTAssertEqual(settings.maxTokens, 512)
        XCTAssertEqual(settings.temperature, 0.4)
        XCTAssertEqual(settings.topP, 0.8)
        XCTAssertEqual(settings.engineOverride, "text-chat-gemma4")
    }

    func testModelReadinessParserFindsInstalledMissingAndUnknown() {
        let output = """
        ID Category Status Size
        image-zimage-nano image installed 4 GB
        video-ltx-av media missing 12 GB
        vision-segment-sam31 vision unsupported 7 GB
        """

        XCTAssertEqual(
            ModelReadinessParser.state(for: "image-zimage-nano", modelListOutput: output),
            .ready
        )
        XCTAssertEqual(
            ModelReadinessParser.state(for: "video-ltx-av", modelListOutput: output),
            .missingModel("video-ltx-av")
        )
        XCTAssertEqual(
            ModelReadinessParser.state(for: "vision-segment-sam31", modelListOutput: output),
            .unsupported("vision-segment-sam31 is listed as unsupported on this Mac.")
        )

        if case .unknown = ModelReadinessParser.state(for: "missing", modelListOutput: output) {
            // expected
        } else {
            XCTFail("Expected unknown readiness for a model absent from the list.")
        }
    }

    func testModelCapabilitiesParserFindsSupportAndMemoryReasons() throws {
        let output = """
        Machine
          processor: M2 Max
          unifiedMemory: 32 GB
          appleSiliconMac: true

        Model capabilities
        - image-zimage-nano [supported]
          title: Image Nano
          category: image
          memory: minimum 12 GB, recommended 16 GB
          download: Hugging Face snapshot
        - image-zimage-max [unsupported]
          title: Image, realistic quality
          category: image
          memory: minimum 48 GB, recommended 64 GB
          download: Hugging Face snapshot
          reason: Requires at least 48 GB unified memory; detected 32 GB.
        """

        let capabilities = ModelCapabilitiesParser.capabilities(from: output)
        let nano = try XCTUnwrap(capabilities["image-zimage-nano"])
        let max = try XCTUnwrap(capabilities["image-zimage-max"])

        XCTAssertTrue(nano.isSupported)
        XCTAssertEqual(nano.minimumUnifiedMemoryGB, 12)
        XCTAssertNil(nano.unavailableMessage)
        XCTAssertFalse(max.isSupported)
        XCTAssertEqual(max.minimumUnifiedMemoryGB, 48)
        XCTAssertEqual(max.recommendedUnifiedMemoryGB, 64)
        XCTAssertEqual(
            max.unavailableMessage,
            "Requires at least 48 GB unified memory; detected 32 GB."
        )
    }

    func testModelCapabilitiesParserReadsJSONRecommendationReport() throws {
        let output = """
        {
          "recommendedChatModel" : {
            "modelID" : "text-chat-q36-nano"
          },
          "recommendedCodeModel" : {
            "id" : "text-code-north-mini"
          },
          "models" : [
            {
              "download" : "hugging-face",
              "id" : "text-chat-q36-nano",
              "minimumUnifiedMemoryGB" : 24,
              "reasons" : [],
              "recommendedUnifiedMemoryGB" : 32,
              "supported" : true
            },
            {
              "download" : "hugging-face",
              "id" : "text-chat-gemma4",
              "minimumUnifiedMemoryGB" : 48,
              "reasons" : [
                "Requires at least 48 GB unified memory; detected 32 GB."
              ],
              "recommendedUnifiedMemoryGB" : 64,
              "supported" : false
            }
          ]
        }
        """

        let report = ModelCapabilitiesParser.report(from: output)
        XCTAssertEqual(report.recommendedChatModelID, "text-chat-q36-nano")
        XCTAssertEqual(report.recommendedCodeModelID, "text-code-north-mini")
        let q36 = try XCTUnwrap(report.capabilitiesByID["text-chat-q36-nano"])
        let gemma = try XCTUnwrap(report.capabilitiesByID["text-chat-gemma4"])

        XCTAssertTrue(q36.isSupported)
        XCTAssertFalse(gemma.isSupported)
        XCTAssertEqual(gemma.minimumUnifiedMemoryGB, 48)
        XCTAssertEqual(
            gemma.unavailableMessage,
            "Requires at least 48 GB unified memory; detected 32 GB."
        )
    }

    func testAPIKeyPreviewMasking() {
        let args = ["api", "serve", "--api-key", "secret-token", "--port", "8080"]
        XCTAssertEqual(
            args.maskingSecrets(),
            ["api", "serve", "--api-key", "••••••••", "--port", "8080"]
        )
    }

    func testAPIKeyLaunchUsesEnvironmentInsteadOfArguments() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        var draft = template.defaultDraft()
        draft.apiKey = " secret-token "

        let args = template.arguments(from: draft)
        let env = CommandLaunchEnvironment.overrides(templateID: template.id, draft: draft)

        XCTAssertFalse(args.contains("--api-key"))
        XCTAssertFalse(args.contains("secret-token"))
        XCTAssertEqual(env["MERERUN_API_KEY"], "secret-token")
    }
}
