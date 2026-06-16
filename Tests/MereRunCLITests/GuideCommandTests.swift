import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class GuideCommandTests: XCTestCase {
    func testGuideListParses() throws {
        let command = try GuideCommand.parse(["--list"])

        XCTAssertTrue(command.list)
        XCTAssertTrue(command.commandPath.isEmpty)
    }

    func testGuideMusicGenerateParsesAndResolves() throws {
        let command = try GuideCommand.parse(["music", "generate"])
        let entry = try GuideCommand.resolveEntry(commandPath: command.commandPath, model: command.model)

        XCTAssertEqual(command.commandPath, ["music", "generate"])
        XCTAssertEqual(entry.topic, "music-generate")
    }

    func testGuideMusicAnalyzeParsesAndResolves() throws {
        let command = try GuideCommand.parse(["music", "analyze"])
        let entry = try GuideCommand.resolveEntry(commandPath: command.commandPath, model: command.model)

        XCTAssertEqual(command.commandPath, ["music", "analyze"])
        XCTAssertEqual(entry.topic, "music-analyze")
    }

    func testGuideMusicGenerateModelFocusParsesAndRenders() throws {
        let command = try GuideCommand.parse([
            "music",
            "generate",
            "--model",
            ModelResolver.ModelID.magentaRT2Small.rawValue,
        ])
        let entry = try GuideCommand.resolveEntry(commandPath: command.commandPath, model: command.model)
        let rendered = try GuideCommand.render(entry: entry, model: command.model, json: false)

        XCTAssertEqual(command.model, ModelResolver.ModelID.magentaRT2Small.rawValue)
        XCTAssertTrue(rendered.contains("Model focus: `\(ModelResolver.ModelID.magentaRT2Small.rawValue)`"))
        XCTAssertTrue(rendered.contains("# Music Generate"))
    }

    func testGuideImageGenerateModelFocusParses() throws {
        let command = try GuideCommand.parse(["image", "generate", "--model", "image-zimage-max"])
        let entry = try GuideCommand.resolveEntry(commandPath: command.commandPath, model: command.model)

        XCTAssertEqual(entry.topic, "image-generate")
        XCTAssertEqual(command.model, "image-zimage-max")
    }

    func testGuideVideoGenerateCoversLTX23UnifiedAVModel() throws {
        let command = try GuideCommand.parse([
            "video",
            "generate",
            "--model",
            ModelResolver.ModelID.ltxVideo23AVMLX.rawValue,
        ])
        let entry = try GuideCommand.resolveEntry(commandPath: command.commandPath, model: command.model)
        let rendered = try GuideCommand.render(entry: entry, model: command.model, json: false)

        XCTAssertEqual(entry.topic, "video-generate")
        XCTAssertEqual(command.model, ModelResolver.ModelID.ltxVideo23AVMLX.rawValue)
        XCTAssertTrue(rendered.contains("Model focus: `\(ModelResolver.ModelID.ltxVideo23AVMLX.rawValue)`"))
        XCTAssertTrue(rendered.contains("--variant unified-av"))
    }

    func testGuideOpenWebUIParsesAndRenders() throws {
        let command = try GuideCommand.parse(["open-webui"])
        let entry = try GuideCommand.resolveEntry(commandPath: command.commandPath, model: command.model)
        let rendered = try GuideCommand.render(entry: entry, model: nil, json: false)

        XCTAssertEqual(entry.topic, "open-webui")
        XCTAssertTrue(rendered.contains("http://host.docker.internal:8080/v1"))
        XCTAssertTrue(rendered.contains("text-embed-qwen3-0.6b"))
        XCTAssertTrue(rendered.contains("scripts/smoke-open-webui.sh live-smoke"))
        XCTAssertTrue(rendered.contains("proxy-smoke"))
        XCTAssertTrue(rendered.contains("OPEN_WEBUI_CHAT_MODELS"))
        XCTAssertTrue(rendered.contains("\"function_calling\":\"native\""))
        XCTAssertTrue(rendered.contains("IMAGES_EDIT_OPENAI_API_BASE_URL"))
    }

    func testUnknownGuideTopicThrows() {
        XCTAssertThrowsError(
            try GuideCommand.resolveEntry(commandPath: ["banana", "split"], model: nil)
        ) { error in
            XCTAssertTrue(String(describing: error).contains("Unknown guide topic"))
        }
    }

    func testUnsupportedModelForKnownGuideThrows() {
        XCTAssertThrowsError(
            try GuideCommand.resolveEntry(commandPath: ["music", "generate"], model: "image-zimage-max")
        ) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("not covered"))
            XCTAssertTrue(message.contains("music-acestep"))
            XCTAssertTrue(message.contains(ModelResolver.ModelID.magentaRT2Small.rawValue))
        }
    }

    func testGuideJSONShapeIsValid() throws {
        let entry = try XCTUnwrap(GuideRegistry.topic(matching: ["music", "generate"]))
        let rendered = try GuideCommand.render(entry: entry, model: nil, json: true)
        let data = try XCTUnwrap(rendered.data(using: .utf8))
        let payload = try JSONDecoder().decode(GuideJSONPayload.self, from: data)

        XCTAssertEqual(payload.topic, "music-generate")
        XCTAssertEqual(payload.title, "Music Generate")
        XCTAssertEqual(payload.commands, ["music generate"])
        XCTAssertEqual(
            payload.models,
            [
                "music-acestep",
                ModelResolver.ModelID.aceStepXLTurbo.rawValue,
                ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue,
                ModelResolver.ModelID.magentaRT2Small.rawValue,
                ModelResolver.ModelID.magentaRT2Base.rawValue,
            ]
        )
        XCTAssertTrue(payload.content.contains("# Music Generate"))
    }

    func testGuideListRendersStableTable() throws {
        let rendered = try GuideCommand.renderList(json: false)

        XCTAssertTrue(rendered.contains("Topic"))
        XCTAssertTrue(rendered.contains("image-generate"))
        XCTAssertTrue(rendered.contains("music generate"))
        XCTAssertTrue(rendered.contains("Read one with: mere.run guide <command path>"))
    }

    func testGuideListRendersMarkdownTableWithDescriptions() throws {
        let rendered = try GuideCommand.renderListMarkdown()

        XCTAssertTrue(rendered.contains("| Topic | Command | Models | Title | Description |"))
        XCTAssertTrue(rendered.contains("| --- | --- | --- | --- | --- |"))
        XCTAssertTrue(rendered.contains("| image-validate |"))
        // Description column is sourced from each guide's Purpose section.
        XCTAssertTrue(rendered.contains("Run deterministic checks against local image model layouts"))
    }

    func testGuidePurposeSummaryNonEmptyForEveryGuide() {
        for entry in GuideRegistry.all {
            XCTAssertFalse(
                GuideRegistry.purposeSummary(for: entry).isEmpty,
                "Guide \(entry.topic) has no Purpose summary for the markdown description column."
            )
        }
    }

    func testGuideRegistryIntegrity() throws {
        XCTAssertFalse(GuideRegistry.all.isEmpty)

        var seenTopics = Set<String>()
        for entry in GuideRegistry.all {
            XCTAssertFalse(entry.topic.isEmpty)
            XCTAssertTrue(seenTopics.insert(entry.topic).inserted, "Duplicate guide topic: \(entry.topic)")
            XCTAssertFalse(entry.title.isEmpty, "Guide \(entry.topic) is missing a title.")
            XCTAssertFalse(entry.commandPaths.isEmpty, "Guide \(entry.topic) is missing command paths.")
            XCTAssertFalse(entry.resourceName.isEmpty, "Guide \(entry.topic) is missing a resource name.")

            let content = try GuideRegistry.content(for: entry)
            XCTAssertTrue(content.contains("# \(entry.title)"), "Guide \(entry.topic) is missing its title heading.")
            XCTAssertTrue(content.contains("## Sources"), "Guide \(entry.topic) is missing a Sources section.")
            XCTAssertFalse(content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

            for model in entry.models {
                XCTAssertNotNil(
                    ManagedModelCatalog.spec(for: model),
                    "Guide \(entry.topic) references unknown managed model \(model)."
                )
            }
        }
    }

    func testTopLevelCommandIncludesGuide() {
        let commandNames = Set(MereRunCLI.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("guide"))
    }
}

private struct GuideJSONPayload: Decodable {
    let topic: String
    let title: String
    let commands: [String]
    let models: [String]
    let content: String
}
