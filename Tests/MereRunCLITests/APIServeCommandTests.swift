import Foundation
import XCTest
import AudioCore
@testable import MereRunCLI
@testable import MereRunCore

final class APIServeCommandTests: XCTestCase {
    func testAPICommandExposesServeSubcommand() {
        let commandNames = Set(API.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(commandNames, Set(["serve"]))
    }

    func testAPIServeParsesDefaults() throws {
        let cmd = try APIServe.parse([])

        XCTAssertEqual(cmd.port, 8080)
        XCTAssertEqual(cmd.host, "127.0.0.1")
        XCTAssertEqual(cmd.engine, .textChatQ36)
        XCTAssertNil(cmd.model)
        XCTAssertNil(cmd.lora)
        XCTAssertNil(cmd.apiKey)
        XCTAssertEqual(cmd.rateLimitPerMinute, 60)
        XCTAssertEqual(cmd.maxActiveRequests, 1)
        XCTAssertEqual(cmd.memoryGuard, .balanced)
        XCTAssertNil(cmd.memoryGuardCustomCeilingGB)
        XCTAssertEqual(cmd.contextSize, 32_768)
        XCTAssertNil(cmd.kvBits)
        XCTAssertNil(cmd.kvQuantScheme)
        XCTAssertNil(cmd.kvGroupSize)
        XCTAssertNil(cmd.quantizedKVStart)
    }

    func testAPIServeParsesOverrides() throws {
        let cmd = try APIServe.parse([
            "--host", "0.0.0.0",
            "--port", "11434",
            "--engine", "text-chat-gemma4",
            "--model-path", "/tmp/gemma4",
            "--lora", "/tmp/adapter.safetensors",
            "--api-key", "secret",
            "--rate-limit-per-minute", "120",
            "--max-active-requests", "2",
            "--memory-guard", "custom",
            "--memory-guard-custom-ceiling-gb", "42.5",
            "--context-size", "8192",
            "--kv-bits", "3.5",
            "--kv-quant-scheme", "turboquant",
            "--kv-group-size", "32",
            "--quantized-kv-start", "2048",
        ])

        XCTAssertEqual(cmd.host, "0.0.0.0")
        XCTAssertEqual(cmd.port, 11_434)
        XCTAssertEqual(cmd.engine, .textChatGemma4)
        XCTAssertEqual(cmd.model, "/tmp/gemma4")
        XCTAssertEqual(cmd.lora, "/tmp/adapter.safetensors")
        XCTAssertEqual(cmd.apiKey, "secret")
        XCTAssertEqual(cmd.rateLimitPerMinute, 120)
        XCTAssertEqual(cmd.maxActiveRequests, 2)
        XCTAssertEqual(cmd.memoryGuard, .custom)
        XCTAssertEqual(cmd.memoryGuardCustomCeilingGB, 42.5)
        XCTAssertEqual(cmd.contextSize, 8192)
        XCTAssertEqual(cmd.kvBits, 3.5)
        XCTAssertEqual(cmd.kvQuantScheme, "turboquant")
        XCTAssertEqual(cmd.kvGroupSize, 32)
        XCTAssertEqual(cmd.quantizedKVStart, 2048)
    }

    func testAPIServeGemma4TurboModelDefaultsToTurboQuantKVCache() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-gemma4",
            "--model-path", Gemma4Resources.turboModelId,
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization()

        XCTAssertEqual(quantization.bits, Gemma4Resources.defaultTurboKVBits)
        XCTAssertEqual(quantization.scheme, .turboquant)
        XCTAssertEqual(quantization.groupSize, Gemma4Resources.defaultKVGroupSize)
        XCTAssertEqual(quantization.quantizedStart, Gemma4Resources.defaultTurboQuantizedKVStart)
    }

    func testAPIServeParsesLFM2Engine() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-lfm2",
            "--model-path", LFM2Resources.defaultModelId,
        ])

        XCTAssertEqual(cmd.engine, .textChatLFM2)
        XCTAssertEqual(cmd.model, LFM2Resources.defaultModelId)
    }

    func testAPIServeParsesLegacyQ35EngineAlias() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-q35",
        ])

        XCTAssertEqual(cmd.engine, .textChatQ35)
        XCTAssertEqual(cmd.engine.runtimeServingEngine, .textChatQ36)
    }

    func testAPIServeGemma4TurboKVFlagsOverrideIndependently() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-gemma4",
            "--model-path", Gemma4Resources.turboModelId,
            "--kv-quant-scheme", "uniform",
            "--kv-group-size", "32",
            "--quantized-kv-start", "128",
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization()

        XCTAssertEqual(quantization.bits, Gemma4Resources.defaultTurboKVBits)
        XCTAssertEqual(quantization.scheme, .uniform)
        XCTAssertEqual(quantization.groupSize, 32)
        XCTAssertEqual(quantization.quantizedStart, 128)
    }

    func testAPIServeGemma4PolarKVFlagsParse() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-gemma4",
            "--model-path", Gemma4Resources.turboModelId,
            "--kv-bits", "2",
            "--kv-quant-scheme", "polar",
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization()

        XCTAssertEqual(quantization.bits, 2)
        XCTAssertEqual(quantization.scheme, .polar)
    }

    func testHealthContractUsesStablePayload() {
        XCTAssertEqual(APIServerContract.healthStatus(), APIHealthStatus(status: "ok"))
    }

    func testModelsContractUsesProvidedModelID() {
        let response = APIServerContract.modelsResponse(
            modelId: "mererun-test-model",
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(response.object, "list")
        XCTAssertEqual(response.data.count, 1)
        XCTAssertEqual(response.data.first?.id, "mererun-test-model")
        XCTAssertEqual(response.data.first?.object, "model")
        XCTAssertEqual(response.data.first?.owned_by, "mere.run")
        XCTAssertEqual(response.data.first?.created, 123)
    }

    func testModelsContractCanReturnAliases() {
        let response = APIServerContract.modelsResponse(
            modelIds: ["text-chat-gemma4", "chat-default"],
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(response.data.map(\.id), ["text-chat-gemma4", "chat-default"])
        XCTAssertEqual(Set(response.data.map(\.owned_by)), Set(["mere.run"]))
    }

    func testEmbeddingRequestDecodesStringInputAndUnknownFields() throws {
        let data = """
        {
          "model": "text-embed-qwen3-0.6b",
          "input": "semantic search query",
          "encoding_format": "float",
          "user": "rag-test",
          "x-client-extra": { "kept": true }
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: data)

        XCTAssertEqual(request.model, "text-embed-qwen3-0.6b")
        XCTAssertEqual(request.input.texts, ["semantic search query"])
        XCTAssertEqual(request.encoding_format, "float")
        XCTAssertEqual(request.user, "rag-test")
        XCTAssertEqual(request.unknownFields["x-client-extra"]?.objectValue?["kept"]?.boolValue, true)
    }

    func testEmbeddingRequestDecodesArrayInput() throws {
        let data = """
        {
          "model": "text-embed-qwen3-0.6b",
          "input": ["first", "second"]
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: data)

        XCTAssertEqual(request.input.texts, ["first", "second"])
    }

    func testEmbeddingContractValidatesSupportedOptions() throws {
        let request = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .array(["first", "second"]),
            encoding_format: "float"
        )

        XCTAssertEqual(try APIServerContract.embeddingTexts(from: request), ["first", "second"])

        let base64Request = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .string("hello"),
            encoding_format: "base64"
        )
        XCTAssertThrowsError(try APIServerContract.embeddingTexts(from: base64Request)) { error in
            XCTAssertTrue(error.localizedDescription.contains("encoding_format"))
        }

        let dimensionsRequest = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .string("hello"),
            dimensions: 128
        )
        XCTAssertThrowsError(try APIServerContract.embeddingTexts(from: dimensionsRequest)) { error in
            XCTAssertTrue(error.localizedDescription.contains("dimensions"))
        }

        let emptyRequest = OpenAIEmbeddingRequest(
            model: "text-embed-qwen3-0.6b",
            input: .array([])
        )
        XCTAssertThrowsError(try APIServerContract.embeddingTexts(from: emptyRequest)) { error in
            XCTAssertTrue(error.localizedDescription.contains("input"))
        }
    }

    func testEmbeddingContractUsesOpenAICompatibleResponseShape() {
        let response = APIServerContract.embeddingResponse(
            modelId: "text-embed-qwen3-0.6b",
            embeddings: [[0.1, 0.2], [0.3, 0.4]],
            tokenCounts: [3, 5]
        )

        XCTAssertEqual(response.object, "list")
        XCTAssertEqual(response.model, "text-embed-qwen3-0.6b")
        XCTAssertEqual(response.data.count, 2)
        XCTAssertEqual(response.data[0].object, "embedding")
        XCTAssertEqual(response.data[0].index, 0)
        XCTAssertEqual(response.data[0].embedding, [0.1, 0.2])
        XCTAssertEqual(response.usage.prompt_tokens, 8)
        XCTAssertEqual(response.usage.total_tokens, 8)
    }

    func testCompanionModelIDsIncludeNativeOpenAIStyleModalities() {
        let ids = APIServerContract.companionModelIDs(installedModelIDs: Set([
            "image-zimage-nano",
            "speech-tts-qwen3-nano",
            "speech-asr-parakeet",
            "text-embed-qwen3-0.6b",
            "qwen-image-edit",
        ]))

        XCTAssertTrue(ids.contains("text-embed-qwen3-0.6b"))
        XCTAssertTrue(ids.contains("image-zimage-nano"))
        XCTAssertTrue(ids.contains("speech-tts-qwen3-nano"))
        XCTAssertTrue(ids.contains("speech-asr-parakeet"))
        XCTAssertTrue(ids.contains("qwen-image-edit"))
        XCTAssertFalse(ids.contains("image-zimage-max"))
        XCTAssertFalse(ids.contains("speech-asr-qwen3"))
    }

    func testCompanionModelIDsHideMissingSidecarModels() {
        let ids = APIServerContract.companionModelIDs(installedModelIDs: [])

        XCTAssertFalse(ids.contains("image-zimage-nano"))
        XCTAssertFalse(ids.contains("speech-tts-qwen3-nano"))
        XCTAssertFalse(ids.contains("speech-asr-parakeet"))
        XCTAssertFalse(ids.contains("text-embed-qwen3-0.6b"))
    }

    func testImageGenerationContractMapsOpenAINamesAndValidatesOptions() throws {
        let request = OpenAIImageGenerationRequest(
            prompt: "  workstation in morning light  ",
            model: "dall-e-3",
            n: 1,
            size: "512x768",
            response_format: "url",
            seed: 123,
            negative_prompt: "blur",
            steps: 4,
            guidance_scale: 1.2
        )

        let plan = try APIServerContract.imageGenerationPlan(from: request)

        XCTAssertEqual(plan.modelID, "image-zimage-nano")
        XCTAssertEqual(plan.prompt, "workstation in morning light")
        XCTAssertEqual(plan.width, 512)
        XCTAssertEqual(plan.height, 768)
        XCTAssertEqual(plan.responseFormat, "url")
        XCTAssertEqual(plan.seed, 123)
        XCTAssertEqual(plan.negativePrompt, "blur")
        XCTAssertEqual(plan.steps, 4)
        XCTAssertEqual(plan.guidanceScale, 1.2)
        XCTAssertNil(plan.inputImage)
        XCTAssertNil(plan.strength)

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", n: 2)
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("n"))
        }

        XCTAssertThrowsError(
            try APIServerContract.imageGenerationPlan(
                from: OpenAIImageGenerationRequest(prompt: "hello", size: "large")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("size"))
        }
    }

    func testImageGenerationResponseCanReturnBase64OrFileURL() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-api-test-\(UUID().uuidString).png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }
        let plan = APIServerContract.ImageGenerationPlan(
            modelID: "image-zimage-nano",
            prompt: "hello",
            width: 1024,
            height: 1024,
            responseFormat: "b64_json",
            seed: nil,
            negativePrompt: nil,
            steps: nil,
            guidanceScale: nil,
            inputImage: nil,
            strength: nil
        )

        let b64 = try APIServerContract.imageResponse(
            outputURL: tempURL,
            plan: plan,
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(b64.created, 123)
        XCTAssertEqual(b64.data.first?.b64_json, Data([0x89, 0x50, 0x4E, 0x47]).base64EncodedString())
        XCTAssertEqual(b64.data.first?.revised_prompt, "hello")

        let urlPlan = APIServerContract.ImageGenerationPlan(
            modelID: plan.modelID,
            prompt: plan.prompt,
            width: plan.width,
            height: plan.height,
            responseFormat: "url",
            seed: nil,
            negativePrompt: nil,
            steps: nil,
            guidanceScale: nil,
            inputImage: nil,
            strength: nil
        )
        let urlResponse = try APIServerContract.imageResponse(outputURL: tempURL, plan: urlPlan)
        XCTAssertEqual(urlResponse.data.first?.url, tempURL.absoluteString)
    }

    func testImageEditContractReadsMultipartFields() throws {
        let inputURL = URL(fileURLWithPath: "/tmp/input.png")
        let secondURL = URL(fileURLWithPath: "/tmp/second.png")
        let maskURL = URL(fileURLWithPath: "/tmp/mask.png")
        let form = MultipartFormData(parts: [
            MultipartFormData.Part(name: "model", filename: nil, contentType: nil, body: Data("gpt-image-1".utf8)),
            MultipartFormData.Part(name: "prompt", filename: nil, contentType: nil, body: Data("  add warm light  ".utf8)),
            MultipartFormData.Part(name: "size", filename: nil, contentType: nil, body: Data("768x512".utf8)),
            MultipartFormData.Part(name: "response_format", filename: nil, contentType: nil, body: Data("url".utf8)),
            MultipartFormData.Part(name: "seed", filename: nil, contentType: nil, body: Data("42".utf8)),
            MultipartFormData.Part(name: "strength", filename: nil, contentType: nil, body: Data("0.4".utf8)),
        ])

        let plan = try APIServerContract.imageEditPlan(
            from: form,
            inputImageURLs: [inputURL, secondURL],
            maskImageURL: maskURL
        )

        XCTAssertEqual(plan.modelID, "image-zimage-nano")
        XCTAssertEqual(plan.prompt, "add warm light")
        XCTAssertEqual(plan.width, 768)
        XCTAssertEqual(plan.height, 512)
        XCTAssertEqual(plan.responseFormat, "url")
        XCTAssertEqual(plan.seed, 42)
        XCTAssertEqual(plan.inputImage, inputURL)
        XCTAssertEqual(plan.additionalInputImages, [secondURL])
        XCTAssertEqual(plan.maskImage, maskURL)
        XCTAssertEqual(plan.strength, 0.4)
    }

    func testSpeechContractMapsOpenAINamesAndValidatesOutputFormats() throws {
        let request = OpenAIAudioSpeechRequest(
            model: "tts-1",
            input: "  hello from mere.run  ",
            voice: "onyx",
            response_format: "mp3",
            speed: 1.25,
            instructions: "Use a friendly pace.",
            temperature: 0.7
        )

        let plan = try APIServerContract.speechPlan(from: request)

        XCTAssertEqual(plan.modelID, "speech-tts-qwen3-nano")
        XCTAssertEqual(plan.input, "hello from mere.run")
        XCTAssertTrue(plan.voiceDescription.contains("deep"))
        XCTAssertTrue(plan.voiceDescription.contains("friendly pace"))
        XCTAssertEqual(plan.responseFormat, "mp3")
        XCTAssertEqual(APIServerContract.speechContentType(for: plan.responseFormat), "audio/mpeg")
        XCTAssertEqual(plan.speed, 1.25)
        XCTAssertEqual(plan.temperature, 0.7)

        XCTAssertThrowsError(
            try APIServerContract.speechPlan(
                from: OpenAIAudioSpeechRequest(input: "hello", response_format: "caf")
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("response_format"))
        }
    }

    func testMultipartFormDataParserExtractsFieldsAndFile() throws {
        let boundary = "mere-boundary"
        let body = [
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"model\"",
            "",
            "whisper-1",
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"",
            "Content-Type: audio/wav",
            "",
            "WAVDATA",
            "--mere-boundary--",
            "",
        ].joined(separator: "\r\n").data(using: .utf8)!

        let form = try MultipartFormData.parse(body: body, boundary: boundary)

        XCTAssertEqual(form.field("model"), "whisper-1")
        XCTAssertEqual(form.file(named: "file")?.filename, "speech.wav")
        XCTAssertEqual(form.files(named: "file").map(\.filename), ["speech.wav"])
        XCTAssertEqual(form.file(named: "file")?.contentType, "audio/wav")
        XCTAssertEqual(form.file(named: "file")?.body, Data("WAVDATA".utf8))
    }

    func testMultipartFormDataParserExtractsRepeatedFiles() throws {
        let boundary = "mere-boundary"
        let body = [
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"image\"; filename=\"first.png\"",
            "Content-Type: image/png",
            "",
            "FIRST",
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"image[]\"; filename=\"second.png\"",
            "Content-Type: image/png",
            "",
            "SECOND",
            "--mere-boundary",
            "Content-Disposition: form-data; name=\"mask\"; filename=\"mask.png\"",
            "Content-Type: image/png",
            "",
            "MASK",
            "--mere-boundary--",
            "",
        ].joined(separator: "\r\n").data(using: .utf8)!

        let form = try MultipartFormData.parse(body: body, boundary: boundary)

        XCTAssertEqual(form.files(named: "image").map(\.filename), ["first.png"])
        XCTAssertEqual(form.files(named: "image[]").map(\.filename), ["second.png"])
        XCTAssertEqual(form.file(named: "mask")?.body, Data("MASK".utf8))
    }

    func testTranscriptionContractMapsOpenAINamesAndVerboseResponse() throws {
        let form = MultipartFormData(parts: [
            MultipartFormData.Part(name: "model", filename: nil, contentType: nil, body: Data("whisper-1".utf8)),
            MultipartFormData.Part(name: "language", filename: nil, contentType: nil, body: Data("en".utf8)),
            MultipartFormData.Part(name: "response_format", filename: nil, contentType: nil, body: Data("verbose_json".utf8)),
            MultipartFormData.Part(name: "task", filename: nil, contentType: nil, body: Data("transcribe".utf8)),
        ])

        let plan = try APIServerContract.transcriptionPlan(from: form)

        XCTAssertEqual(plan.modelID, "speech-asr-parakeet")
        XCTAssertEqual(plan.language, "en")
        XCTAssertEqual(plan.responseFormat, "verbose_json")
        XCTAssertEqual(plan.task, .transcribe)
        XCTAssertEqual(plan.maxTokens, 448)

        let response = APIServerContract.transcriptionResponse(
            from: ASRResult(
                text: "hello",
                language: "en",
                duration: 1.5,
                sentenceAlignments: [
                    ASRSentenceAlignment(
                        text: "hello",
                        startSeconds: 0,
                        durationSeconds: 1.5,
                        tokens: []
                    ),
                ]
            ),
            verbose: true
        )

        XCTAssertEqual(response.text, "hello")
        XCTAssertEqual(response.language, "en")
        XCTAssertEqual(response.duration, 1.5)
        XCTAssertEqual(response.segments?.first?.text, "hello")
    }

    func testTranscriptionContractCanRenderSubtitleFormats() throws {
        let srtForm = MultipartFormData(parts: [
            MultipartFormData.Part(name: "response_format", filename: nil, contentType: nil, body: Data("srt".utf8)),
        ])
        let plan = try APIServerContract.transcriptionPlan(from: srtForm)
        XCTAssertEqual(plan.responseFormat, "srt")

        let result = ASRResult(
            text: "hello",
            language: "en",
            duration: 1.5,
            sentenceAlignments: [
                ASRSentenceAlignment(text: "hello", startSeconds: 0, durationSeconds: 1.5, tokens: []),
            ]
        )
        let srt = APIServerContract.transcriptionSubtitle(from: result, format: "srt")
        XCTAssertTrue(srt.contains("1\n00:00:00,000 --> 00:00:01,500\nhello"))

        let vtt = APIServerContract.transcriptionSubtitle(from: result, format: "vtt")
        XCTAssertTrue(vtt.hasPrefix("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nhello"))
    }

    func testChatRequestValidationAcceptsBoundedDefaults() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")]
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: " /tmp/default.safetensors ",
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.maxTokens, APIServerContract.defaultMaxTokens)
        XCTAssertEqual(chatRequest.maxContextTokens, 4_096)
        XCTAssertEqual(chatRequest.temperature, 1.0)
        XCTAssertEqual(chatRequest.topP, 0.95)
        XCTAssertFalse(chatRequest.showThinking)
        XCTAssertEqual(chatRequest.maxContextTokens, 4_096)
        XCTAssertEqual(chatRequest.messages, [ChatMessage(role: .user, content: "hello")])
        guard case .local(let path, let scale) = chatRequest.lora else {
            return XCTFail("Expected server-configured LoRA to be applied")
        }
        XCTAssertEqual(path, "/tmp/default.safetensors")
        XCTAssertEqual(scale, 1.0)
    }

    func testChatRequestDecodesStructuredOpenAIContentParts() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            {
              "role": "user",
              "content": [
                { "type": "text", "text": "hello" },
                { "type": "input_text", "text": "setup mere.run" }
              ]
            },
            {
              "role": "assistant"
            }
          ]
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.messages[0].content, "hello\nsetup mere.run")
        XCTAssertEqual(chatRequest.messages[1].content, "")
    }

    func testChatRequestDecodesModernOpenAIFieldsAndUnknownFields() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            { "role": "developer", "content": "Follow repo rules." },
            { "role": "user", "content": "hello" }
          ],
          "max_completion_tokens": 77,
          "stream_options": { "include_usage": true },
          "parallel_tool_calls": true,
          "metadata": { "client": "test" },
          "reasoning_effort": "low",
          "x-client-extra": { "kept": true }
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        XCTAssertEqual(request.max_completion_tokens, 77)
        XCTAssertEqual(request.stream_options?.include_usage, true)
        XCTAssertEqual(request.parallel_tool_calls, true)
        XCTAssertEqual(request.metadata?["client"], "test")
        XCTAssertEqual(request.reasoning_effort, "low")
        XCTAssertEqual(request.unknownFields["x-client-extra"]?.objectValue?["kept"]?.boolValue, true)
    }

    func testChatRequestMapsDeveloperRoleAndMaxCompletionTokens() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [
                OpenAIChatMessage(role: "developer", content: "Follow repo rules."),
                OpenAIChatMessage(role: "user", content: "hello"),
            ],
            max_completion_tokens: 77
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.maxTokens, 77)
        XCTAssertEqual(chatRequest.messages[0].role, .system)
        XCTAssertEqual(chatRequest.messages[0].content, "Follow repo rules.")
    }

    func testChatRequestMapsSupportedStopSequences() throws {
        let single = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stop: .string("END")
        )
        let singleRequest = try APIServerContract.chatRequest(
            from: single,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textCode.openAICompatibility
        )
        XCTAssertEqual(singleRequest.stopSequences, ["END"])

        let multiple = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stop: .array(["END", "", "\nif __name__"])
        )
        let multipleRequest = try APIServerContract.chatRequest(
            from: multiple,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textCode.openAICompatibility
        )
        XCTAssertEqual(multipleRequest.stopSequences, ["END", "\nif __name__"])
    }

    func testChatRequestValidatesImageContentPartsAgainstEngineCapability() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            {
              "role": "user",
              "content": [
                { "type": "text", "text": "describe" },
                { "type": "image_url", "image_url": { "url": "file:///tmp/image.png" } }
              ]
            }
          ]
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("image content"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithToolsAndVision
        )
        XCTAssertEqual(chatRequest.messages[0].content, "describe")
        XCTAssertEqual(chatRequest.messages[0].imageUrl, "file:///tmp/image.png")
    }

    func testChatRequestMapsSupportedOpenAITools() throws {
        let tool = OpenAIChatTool(
            function: OpenAIChatToolFunction(
                name: "lookup",
                description: "Look up a value.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object([
                            "type": .string("string"),
                            "description": .string("Search query"),
                        ]),
                    ]),
                    "required": .array([.string("query")]),
                ])
            )
        )
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            tools: [tool],
            tool_choice: .mode("auto")
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("tools"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithTools
        )
        XCTAssertEqual(chatRequest.tools?.first?.name, "lookup")
        XCTAssertEqual(chatRequest.tools?.first?.parameters["query"]?.description, "Search query")
        XCTAssertEqual(chatRequest.tools?.first?.required, ["query"])
    }

    func testChatRequestAcceptsSpecificFunctionToolChoice() throws {
        let lookup = OpenAIChatTool(
            function: OpenAIChatToolFunction(
                name: "lookup",
                description: "Look up a value.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object(["type": .string("string")]),
                    ]),
                ])
            )
        )
        let summarize = OpenAIChatTool(
            function: OpenAIChatToolFunction(
                name: "summarize",
                description: "Summarize text.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "text": .object(["type": .string("string")]),
                    ]),
                ])
            )
        )
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            tools: [lookup, summarize],
            tool_choice: .function(name: "summarize")
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithTools
        )

        XCTAssertEqual(chatRequest.tools?.map(\.name), ["summarize"])

        let missing = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            tools: [lookup],
            tool_choice: .function(name: "missing")
        )
        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: missing,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localTextWithTools
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("tool_choice"))
        }
    }

    func testChatRequestValidatesStructuredOutputsAgainstEngineCapability() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            response_format: OpenAIResponseFormat(type: "json_object")
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("response_format"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithStructuredJSON
        )
        XCTAssertTrue(chatRequest.requiresJSON)
    }

    func testGemma4EngineCapabilityAcceptsJSONMode() throws {
        let request = OpenAIChatRequest(
            model: "text-chat-gemma4-12b-4bit",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            response_format: OpenAIResponseFormat(type: "json_object")
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textChatGemma4.openAICompatibility
        )
        XCTAssertTrue(chatRequest.requiresJSON)
    }

    func testChatRequestRejectsUnsupportedHighImpactFields() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stop: .string("END"),
            seed: 42,
            presence_penalty: 0.5,
            logprobs: true,
            reasoning_effort: "low",
            think: true
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            let message = error.localizedDescription
            XCTAssertTrue(
                message.contains("stop")
                    || message.contains("seed")
                    || message.contains("presence_penalty")
                    || message.contains("logprobs")
                    || message.contains("reasoning_effort")
                    || message.contains("thinking")
            )
        }
    }

    func testStreamingUsageOptionHonorsCapabilities() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stream_options: OpenAIStreamOptions(include_usage: true)
        )

        XCTAssertTrue(
            try APIServerContract.includeUsageInStreaming(
                request,
                capabilities: .localText
            )
        )

        XCTAssertThrowsError(
            try APIServerContract.includeUsageInStreaming(
                request,
                capabilities: APIEngineCapabilities(supportsUsageInStreaming: false)
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("stream_options.include_usage"))
        }
    }

    func testChatRequestValidationRejectsOversizedMaxTokens() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            max_tokens: 4_097
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("max_tokens"))
        }
    }

    func testChatRequestValidationRejectsPerRequestLora() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            lora: "/tmp/request-controlled.safetensors"
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("lora"))
        }
    }

    func testChatRequestValidationRejectsInvalidSamplingValues() {
        let badTemperature = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            temperature: 3.0
        )
        let badTopP = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            top_p: 1.5
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: badTemperature, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("temperature"))
        }
        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: badTopP, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("top_p"))
        }
    }

    func testChatContentTypeValidationRejectsBrowserSimpleTypes() {
        XCTAssertFalse(APIServerContract.acceptsJSONContentType(nil))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType(""))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("text/plain"))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("application/x-www-form-urlencoded"))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("multipart/form-data; boundary=abc"))
    }

    func testChatContentTypeValidationAcceptsJSONTypes() {
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/json"))
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/json; charset=utf-8"))
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/vnd.openai+json"))
    }

    func testStreamingStatusMessagesAreNotTreatedAsTokens() {
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Generating..."))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Generating response"))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Retrying generation"))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("DS4 chat completion"))
        XCTAssertFalse(APIServerContract.isStreamingStatusMessage("Actual generated text"))
    }

    func testOpenAIUsageReportsPromptAndCompletionTokenCounts() {
        let result = ChatResponse(
            response: "hello",
            tokensGenerated: 128,
            promptTokens: 1039
        )

        let usage = CodeGenServer.openAIUsage(for: result)

        XCTAssertEqual(usage.prompt_tokens, 1039)
        XCTAssertEqual(usage.completion_tokens, 128)
        XCTAssertEqual(usage.total_tokens, 1167)
    }

    func testOpenAIUsageFallsBackToZeroPromptTokensWhenUnreported() {
        let result = ChatResponse(response: "hello", tokensGenerated: 7)

        let usage = CodeGenServer.openAIUsage(for: result)

        XCTAssertEqual(usage.prompt_tokens, 0)
        XCTAssertEqual(usage.completion_tokens, 7)
        XCTAssertEqual(usage.total_tokens, 7)
    }
}
