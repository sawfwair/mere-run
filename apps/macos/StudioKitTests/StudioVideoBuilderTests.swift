@testable import StudioKit
import MereRunContract
import XCTest

/// The video templates send only what the selected runtime family uses, with values it accepts,
/// so the CLI's gate finds nothing to refuse or warn about in any draft Studio can build.
final class StudioVideoBuilderTests: XCTestCase {
    private func maximalDraft(_ id: CommandTemplateID, model: String) throws -> CommandDraft {
        let template = try XCTUnwrap(CommandCatalog.template(id: id))
        return try CommandDraftProbes.maximalDraft(
            from: template.defaultDraft(), booleans: true,
            overriding: ["model": model, "modelRoot": "", "h3ReferenceInputs": ["image:a.png"]]
        )
    }

    private func report(
        _ capability: MereRunCommandCapability, _ arguments: [String]
    ) -> MereRunFamilyResolutionReport {
        let leaf = Array(arguments.dropFirst(capability.command.count))
        return capability.resolutionReport(MereRunCommandInvocation(capability: capability, arguments: leaf))
    }

    func testEveryVideoFamilyGetsAnArgvItsGatePassesWithoutWarnings() throws {
        let cases: [(CommandTemplateID, MereRunCommandCapability)] = [
            (.videoGenerate, MereRunCapabilityCatalog.videoGenerate),
            (.videoRetake, MereRunCapabilityCatalog.videoRetake)
        ]
        for (id, capability) in cases {
            // Families reached only through what is installed, or through a flag Studio doesn't
            // set (FastH3 with an explicit adapter), have no draft of their own here.
            for family in try XCTUnwrap(capability.routing?.families) where family.selectors.allSatisfy(\.absent) {
                guard let model = family.models.first else { continue }
                let arguments = try XCTUnwrap(CommandCatalog.template(id: id))
                    .arguments(from: try maximalDraft(id, model: model))
                let result = report(capability, arguments)
                XCTAssertEqual(result.family, family.id, "\(model)")
                XCTAssertEqual(result.violations, [], "\(model): \(arguments)")
                XCTAssertEqual(result.warnings, [], "\(model): \(arguments)")
            }
        }
    }

    func testBuildersKeepWhatEachFamilyUses() throws {
        func arguments(_ model: String, _ edit: (inout CommandDraft) -> Void = { _ in }) throws -> [String] {
            var draft = try XCTUnwrap(CommandCatalog.template(id: .videoGenerate)).defaultDraft()
            draft.model = model
            draft.inputPath = "start.png"
            draft.endImagePath = "end.png"
            edit(&draft)
            return try XCTUnwrap(CommandCatalog.template(id: .videoGenerate)).arguments(from: draft)
        }
        // The draft checkpoint no longer receives Studio's final-quality default.
        XCTAssertFalse(try arguments("video-ltx23-av-mlx").contains("--quality"))
        XCTAssertEqual(try arguments("video-ltx25-full-bf16").split(separator: "--quality").count, 2)
        // FL2VA keeps its last keyframe; Ref2VA and FastH3 take no keyframes.
        XCTAssertTrue(try arguments("video-minimax-h3-fl2va-bf16-mlx").contains("--end-image"))
        XCTAssertFalse(try arguments("video-minimax-h3-ref2va-mlx").contains("--image"))
        XCTAssertFalse(try arguments("video-minimax-h3-fasth3-vsa-datafree-mlx").contains("--end-image"))
        // Wan takes the start image but not its strength, and no end keyframe.
        let wan = try arguments("video-wan22-ti2v-5b-mlx")
        XCTAssertTrue(wan.contains("--image") && wan.contains("--steps"))
        XCTAssertFalse(wan.contains("--image-strength") || wan.contains("--end-image"))
        // LTX reads no step count, and the non-audio checkpoints drop source audio.
        let distilled = try arguments("video-ltx25-distilled-bf16") { $0.audioPath = "song.wav" }
        XCTAssertFalse(distilled.contains("--steps") || distilled.contains("--audio"))
        // A local folder keeps the full surface; the CLI identifies it when it runs.
        var folder = try XCTUnwrap(CommandCatalog.template(id: .videoGenerate)).defaultDraft()
        folder.model = "/Volumes/models/h3"
        XCTAssertNil(StudioOptionScope.videoGenerate(folder, source: StudioScopeSource(identities: StudioFixedModelIdentities())).family)
    }

    func testExcludedModelsAreRefusedWithTheGatesReason() throws {
        var draft = try XCTUnwrap(CommandCatalog.template(id: .videoGenerate)).defaultDraft()
        draft.model = "video-cosmos3-edge-mlx"
        let reason = try XCTUnwrap(MereRunCapabilityCatalog.videoGenerate.routing?.excludedModel(id: draft.model)).reason
        XCTAssertEqual(
            CommandCatalog.videoValidationMessage(for: .videoGenerate, draft: draft),
            "video-cosmos3-edge-mlx can't run video generate: \(reason)"
        )
    }
}
