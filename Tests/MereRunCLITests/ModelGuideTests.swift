import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ModelGuideTests: XCTestCase {
    func testEveryManagedModelHasExactlyOneBundledHandbook() throws {
        let guides = try ModelGuideRegistry.all()
        let modelIDs = guides.flatMap(\.models)
        XCTAssertEqual(Set(modelIDs), Set(ManagedModelCatalog.allSpecs.map(\.id)))
        XCTAssertEqual(modelIDs.count, Set(modelIDs).count, "A model maps to multiple handbooks")
        XCTAssertEqual(guides.count, Set(guides.map(\.topic)).count)
        XCTAssertEqual(guides.count, Set(guides.map(\.resourceName)).count)
        for guide in guides {
            XCTAssertFalse(guide.models.isEmpty)
            let content = try GuideRegistry.content(for: guide.entry)
            XCTAssertTrue(content.hasPrefix("# \(guide.title)\n"))
            for heading in ["## Example to adapt", "## Controls and variants", "## Sources and validation"] {
                XCTAssertTrue(content.contains(heading), "\(guide.topic) lacks \(heading)")
            }
            for model in guide.models {
                XCTAssertTrue(content.contains("`\(model)`"))
                XCTAssertEqual(try ModelGuideRegistry.guide(for: model), guide)
            }
        }
    }

    func testStandaloneModelSelectionAndListParse() throws {
        let command = try GuideCommand.parse(["--model", "image-klein-9b", "--json"])
        XCTAssertTrue(command.commandPath.isEmpty)
        XCTAssertEqual(command.model, "image-klein-9b")
        XCTAssertTrue(command.json)
        XCTAssertTrue(try GuideCommand.parse(["--list-models"]).listModels)
    }

    func testModelSelectionNormalizesAndRejectsUnknownIDs() throws {
        XCTAssertEqual(
            try ModelGuideRegistry.guide(for: " IMAGE-KLEIN-9B "),
            try ModelGuideRegistry.guide(for: "image-klein-9b")
        )
        XCTAssertThrowsError(try ModelGuideRegistry.guide(for: ""))
        XCTAssertThrowsError(try ModelGuideRegistry.guide(for: "unknown-model"))
    }

    func testListAndContentJSONSupportStudioWithoutCommandPaths() throws {
        let json = try ModelGuideRegistry.renderList(json: true, markdown: false)
        let guides = try JSONDecoder().decode([ModelGuide].self, from: Data(json.utf8))
        XCTAssertEqual(guides, try ModelGuideRegistry.all())
        let ref = try ModelGuideRegistry.guide(for: "video-minimax-h3-ref2va-mlx")
        let base = try ModelGuideRegistry.guide(for: "video-minimax-h3-fl2va-mlx")
        XCTAssertNotEqual(ref.topic, base.topic, "Reference and keyframe workflows need distinct guidance")
        struct Payload: Decodable { let commands: [String]; let models: [String]; let content: String }
        let rendered = try GuideCommand.render(entry: ref.entry, model: ref.models[0], json: true)
        let payload = try JSONDecoder().decode(Payload.self, from: Data(rendered.utf8))
        XCTAssertEqual(payload.commands, [])
        XCTAssertEqual(payload.models, ref.models)
        XCTAssertTrue(payload.content.contains("--reference"))
        XCTAssertTrue(payload.content.contains("have not been validated with model inference"))
    }
}
