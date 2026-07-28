import XCTest
@testable import AudioSTT
@testable import AudioTTS
@testable import MereRunCLI
@testable import MereRunCore

final class OpenWebUICommandTests: XCTestCase {
    func testOpenWebUICommandExposesQuickstart() {
        let commandNames = Set(OpenWebUI.configuration.subcommands.map { $0.configuration.commandName })

        XCTAssertEqual(commandNames, Set(["quickstart"]))
    }

    func testQuickstartParsesDefaults() throws {
        let command = try OpenWebUIQuickstart.parse([])

        XCTAssertEqual(command.host, "0.0.0.0")
        XCTAssertEqual(command.port, 8080)
        XCTAssertNil(command.engine)
        XCTAssertEqual(command.webuiHost, "127.0.0.1")
        XCTAssertEqual(command.webuiPort, 3000)
        XCTAssertEqual(command.containerName, "open-webui-mere-run")
        XCTAssertEqual(command.volumeName, "open-webui-mere-run")
        XCTAssertEqual(command.image, "ghcr.io/open-webui/open-webui:main")
        XCTAssertNil(command.apiKey)
        XCTAssertEqual(command.textModel, Gemma4Resources.twelveBModelId)
        XCTAssertEqual(command.visionModel, Gemma4Resources.visionTwelveBModelId)
        XCTAssertEqual(command.embeddingModel, Qwen3EmbeddingCatalog.modelId)
        XCTAssertEqual(command.imageModel, ModelResolver.ModelID.zetaNano.rawValue)
        XCTAssertEqual(command.ttsModel, Qwen3TTSResources.defaultModelId)
        XCTAssertEqual(command.sttModel, ParakeetResources.defaultModelId)
        XCTAssertEqual(command.ttsFormat, "wav")
        XCTAssertNil(command.adminPassword)
        XCTAssertFalse(command.pull)
        XCTAssertFalse(command.acceptModelLicense)
        XCTAssertFalse(command.skipServer)
        XCTAssertFalse(command.skipDocker)
        XCTAssertFalse(command.skipConfigure)
        XCTAssertFalse(command.reset)
        XCTAssertFalse(command.dryRun)
    }

    func testQuickstartParsesOverrides() throws {
        let command = try OpenWebUIQuickstart.parse([
            "--host", "127.0.0.1",
            "--port", "11434",
            "--engine", "text-chat-lfm2",
            "--webui-host", "0.0.0.0",
            "--webui-port", "3001",
            "--container-name", "owui-test",
            "--volume-name", "owui-volume",
            "--image", "ghcr.io/open-webui/open-webui:latest",
            "--api-key", "secret",
            "--text-model", LFM2Resources.defaultModelId,
            "--vision-model", Gemma4Resources.visionTwelveBModelId,
            "--embedding-model", Qwen3EmbeddingCatalog.modelId,
            "--image-model", "image-zimage-base",
            "--tts-model", "speech-tts-qwen3-customvoice",
            "--stt-model", "speech-asr-qwen3",
            "--tts-format", "mp3",
            "--admin-email", "owner@example.test",
            "--admin-password", "password",
            "--wait-seconds", "30",
            "--pull",
            "--accept-model-license",
            "--skip-server",
            "--skip-docker",
            "--skip-configure",
            "--reset",
            "--dry-run",
            "--quiet",
        ])

        XCTAssertEqual(command.host, "127.0.0.1")
        XCTAssertEqual(command.port, 11_434)
        XCTAssertEqual(command.engine, .textChatLFM2)
        XCTAssertEqual(command.webuiHost, "0.0.0.0")
        XCTAssertEqual(command.webuiPort, 3001)
        XCTAssertEqual(command.containerName, "owui-test")
        XCTAssertEqual(command.volumeName, "owui-volume")
        XCTAssertEqual(command.image, "ghcr.io/open-webui/open-webui:latest")
        XCTAssertEqual(command.apiKey, "secret")
        XCTAssertEqual(command.textModel, LFM2Resources.defaultModelId)
        XCTAssertEqual(command.imageModel, "image-zimage-base")
        XCTAssertEqual(command.ttsModel, "speech-tts-qwen3-customvoice")
        XCTAssertEqual(command.sttModel, "speech-asr-qwen3")
        XCTAssertEqual(command.ttsFormat, "mp3")
        XCTAssertEqual(command.adminEmail, "owner@example.test")
        XCTAssertEqual(command.adminPassword, "password")
        XCTAssertEqual(command.waitSeconds, 30)
        XCTAssertTrue(command.pull)
        XCTAssertTrue(command.acceptModelLicense)
        XCTAssertTrue(command.skipServer)
        XCTAssertTrue(command.skipDocker)
        XCTAssertTrue(command.skipConfigure)
        XCTAssertTrue(command.reset)
        XCTAssertTrue(command.dryRun)
        XCTAssertTrue(command.quiet)
    }

    func testQuickstartRequiresTermsAcknowledgementBeforeItsDefaultRestrictedImagePull() throws {
        let modelsRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-open-webui-terms-\(UUID().uuidString)", isDirectory: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: modelsRoot)
        }
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let blocked = try OpenWebUIQuickstart.parse(["--pull"])
        let message = try XCTUnwrap(blocked.unacknowledgedUsageTermsMessage())
        XCTAssertTrue(message.contains("image-zimage-nano"))
        XCTAssertTrue(message.contains("--accept-model-license"))

        let accepted = try OpenWebUIQuickstart.parse(["--pull", "--accept-model-license"])
        XCTAssertNil(accepted.unacknowledgedUsageTermsMessage())
    }

    func testQuickstartDryRunPlanUsesDockerBridgeAndChatFilter() throws {
        let command = try OpenWebUIQuickstart.parse([
            "--dry-run",
            "--api-key", "abc123",
        ])

        let plan = try command.makePlan(apiKey: "abc123").render()

        XCTAssertTrue(plan.contains("export MERERUN_API_KEY=abc123"))
        XCTAssertTrue(plan.contains("api serve --engine text-chat-gemma4"))
        XCTAssertTrue(plan.contains("--model text-chat-gemma4-12b"))
        XCTAssertTrue(plan.contains("OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1"))
        XCTAssertTrue(plan.contains("RAG_EMBEDDING_MODEL=text-embed-qwen3-0.6b"))
        XCTAssertTrue(plan.contains("ENABLE_IMAGE_EDIT=False"))
        XCTAssertTrue(plan.contains("AUDIO_TTS_OPENAI_PARAMS={\"response_format\":\"wav\"}"))
        XCTAssertTrue(plan.contains("open-webui quickstart --skip-server --skip-docker"))
        XCTAssertTrue(plan.contains("--vision-model vision-chat-gemma4-12b"))
    }

    func testResolvedAPIKeyPrefersExplicitThenEnvironmentThenLocalDefault() {
        XCTAssertEqual(
            OpenWebUIQuickstart.resolvedAPIKey(explicit: " explicit ", environment: ["MERERUN_API_KEY": "env"]),
            "explicit"
        )
        XCTAssertEqual(
            OpenWebUIQuickstart.resolvedAPIKey(explicit: nil, environment: ["MERERUN_API_KEY": " env "]),
            "env"
        )
        XCTAssertEqual(
            OpenWebUIQuickstart.resolvedAPIKey(explicit: "", environment: [:]),
            "change-me"
        )
    }

    func testResolvedAdminPasswordPrefersExplicitThenEnvironmentThenLocalDefault() {
        XCTAssertEqual(
            OpenWebUIQuickstart.resolvedAdminPassword(
                explicit: " explicit ",
                environment: [OpenWebUIQuickstart.adminPasswordEnvironmentKey: "env"]
            ),
            "explicit"
        )
        XCTAssertEqual(
            OpenWebUIQuickstart.resolvedAdminPassword(
                explicit: nil,
                environment: [OpenWebUIQuickstart.adminPasswordEnvironmentKey: " env "]
            ),
            "env"
        )
        XCTAssertEqual(
            OpenWebUIQuickstart.resolvedAdminPassword(explicit: "", environment: [:]),
            "admin"
        )
    }
}
