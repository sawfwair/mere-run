@testable import MereRunApp
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
        XCTAssertEqual(codeTemplate.defaultModel, "text-code-qwen3")

        var draft = StudioDraft()
        draft.reset(for: .code)
        XCTAssertEqual(draft.model, "text-code-qwen3")
        XCTAssertEqual(StudioCommandAdapter.requiredModel(for: .code, draft: draft), "text-code-qwen3")
    }

    func testChatStudioDefaultsToStreamingOutput() throws {
        let chatTemplate = try XCTUnwrap(CommandCatalog.template(id: .textChat))
        let draft = chatTemplate.defaultDraft()

        XCTAssertTrue(draft.stream)
        XCTAssertTrue(chatTemplate.arguments(from: draft).contains("--stream"))
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
                "--data", "/tmp/krea-dataset",
                "--output", "/tmp/krea-style.safetensors",
                "--width", "768",
                "--height", "768",
                "--training-steps", "1200",
                "--model", "image-krea2-raw",
                "--seed", "7",
            ]
        )
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
        """

        let rows = StudioModelInventoryParser.rows(from: output)

        XCTAssertEqual(rows.count, 3)
        XCTAssertEqual(rows[0].id, "image-zimage-nano")
        XCTAssertEqual(rows[0].status, "installed")
        XCTAssertEqual(rows[0].size, "4.2 GB")
        XCTAssertTrue(rows[2].isInstalled)
        XCTAssertFalse(rows[1].isInstalled)
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
