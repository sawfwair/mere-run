import Foundation
import MereRunContract
import XCTest
@testable import MereRunCore

final class ManagedGemma4A4BBF16Tests: XCTestCase {
    func testPinnedBF16EntryIsDistinctFromTheNVFP4AndDenseModels() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4Resources.a4bBF16UpstreamModelId))
        XCTAssertEqual(spec.id, ModelResolver.ModelID.gemma4A4BBF16.rawValue)
        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.validationKind, .gemma4)
        XCTAssertEqual(spec.hubFallback?.revision, "e13fae2a81ec07e3092a3ebb70c80970b640dc3c")
        XCTAssertEqual(spec.hubFallback?.patterns, Gemma4Resources.snapshotPatterns)
        XCTAssertEqual(spec.estimatedDownloadBytes, 51_644_318_400)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertNotNil(spec.apiProfile)
        XCTAssertNotEqual(spec.id, Gemma4Resources.turboModelId)
        XCTAssertNotEqual(spec.upstreamRepoId, Gemma4Resources.maxUpstreamModelId)

        let manifest = MereRunModelManifest.template(for: .gemma4A4BBF16)
        XCTAssertEqual(manifest.id, spec.id)
        XCTAssertEqual(manifest.engine, .gemma4)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertNil(manifest.quantization)
        XCTAssertEqual(manifest.supports, [.chat])
    }

    func testBF16UsesTheNativeTextPolicyWithoutTurboOrVisionDefaults() {
        let id = Gemma4Resources.a4bBF16ModelId
        XCTAssertTrue(Gemma4Resources.handles(modelSpec: id))
        XCTAssertFalse(Gemma4Resources.usesTurboDefaults(modelSpec: id))
        XCTAssertFalse(Gemma4Resources.supportsVision(modelSpec: id))
        XCTAssertFalse(Gemma4Resources.usesTurboDefaults(modelSpec: Gemma4Resources.a4bBF16UpstreamModelId))
        XCTAssertTrue(Gemma4Resources.usesTurboDefaults(modelSpec: Gemma4Resources.turboModelId))
    }

    func testChatAndTrainingRouteToTheExistingGemmaFamily() throws {
        for command in ["text.chat", "text.train-lora"] {
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: command))
            let family = try XCTUnwrap(capability.routing?.families.first {
                $0.models.contains(Gemma4Resources.a4bBF16ModelId)
            })
            XCTAssertEqual(family.id, "gemma4")
        }
    }
}
