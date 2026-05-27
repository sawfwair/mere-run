import Foundation
import XCTest
@testable import MereRunCore

final class MereRunModelManifestTests: MereRunCoreTestCase {

    func testTemplateRoundTrip() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let manifest = MereRunModelManifest.template(for: .kleinNano, createdAt: Date(timeIntervalSince1970: 0))
        try manifest.write(to: temp)

        let loaded = try MereRunModelManifest.loadIfPresent(from: temp)
        XCTAssertNotNil(loaded)
        guard let loaded else { return }
        XCTAssertEqual(loaded.id, "image-klein-nano")
        XCTAssertEqual(loaded.engine, .flux2Klein)
        XCTAssertEqual(loaded.family, .klein)
        XCTAssertEqual(loaded.tier, .nano)
        XCTAssertEqual(loaded.variant, .distilled)
        XCTAssertEqual(loaded.precision, .int4)
        XCTAssertEqual(loaded.quantization?.bits, 4)
        XCTAssertEqual(loaded.quantization?.groupSize, 64)
        XCTAssertEqual(loaded.defaults?.steps, 4)
        XCTAssertEqual(loaded.supports?.contains(.referenceEdit), true)
    }

    func testWriteTemplateIfKnownOverwritesInvalidJSON() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        // Write a broken manifest file.
        let url = MereRunModelManifest.url(in: temp)
        try TestFileSystem.writeFile(url, contents: Data("{not-json".utf8))

        let written = try MereRunModelManifest.writeTemplateIfKnown(modelId: "image-zimage-max", to: temp, createdAt: Date(timeIntervalSince1970: 0))
        XCTAssertNotNil(written)
        guard let written else { return }
        XCTAssertEqual(written.id, "image-zimage-max")

        let loaded = try MereRunModelManifest.loadIfPresent(from: temp)
        XCTAssertNotNil(loaded)
        guard let loaded else { return }
        XCTAssertEqual(loaded.id, "image-zimage-max")
        XCTAssertEqual(loaded.engine, .zimageTurbo)
    }

    func testZImageNanoTemplateUsesMFluxUpstream() throws {
        let manifest = MereRunModelManifest.template(for: .zetaNano, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-zimage-nano")
        XCTAssertEqual(manifest.engine, .zimageTurbo)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.upstreamRepoId, "filipstrand/Z-Image-Turbo-mflux-4bit@main")
    }

    func testBonsaiTernaryTemplateHasExpectedNativeLayout() throws {
        let manifest = MereRunModelManifest.template(for: .bonsaiTernary, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-bonsai-ternary")
        XCTAssertEqual(manifest.engine, .flux2Klein)
        XCTAssertEqual(manifest.family, .klein)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .int2)
        XCTAssertEqual(manifest.quantization?.bits, 2)
        XCTAssertEqual(manifest.quantization?.groupSize, 128)
        XCTAssertEqual(manifest.defaults?.steps, 4)
        XCTAssertEqual(manifest.defaults?.cfg, 1.0)
        XCTAssertEqual(manifest.defaults?.sigmaShift, 3.0)
        XCTAssertEqual(manifest.components?.textEncoder, .local(path: "text_encoder-mlx-4bit"))
        XCTAssertEqual(manifest.components?.transformer, .local(path: "transformer-packed-mflux"))
        XCTAssertEqual(manifest.upstreamRepoId, "prism-ml/bonsai-image-ternary-4B-mlx-2bit@main")
    }

    func testBonsaiBinaryTemplateHasExpectedNativeLayout() throws {
        let manifest = MereRunModelManifest.template(for: .bonsaiBinary, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-bonsai-binary")
        XCTAssertEqual(manifest.engine, .flux2Klein)
        XCTAssertEqual(manifest.family, .klein)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .int1)
        XCTAssertEqual(manifest.quantization?.bits, 1)
        XCTAssertEqual(manifest.quantization?.groupSize, 128)
        XCTAssertEqual(manifest.defaults?.steps, 4)
        XCTAssertEqual(manifest.defaults?.cfg, 1.0)
        XCTAssertEqual(manifest.defaults?.sigmaShift, 3.0)
        XCTAssertEqual(manifest.components?.textEncoder, .local(path: "text_encoder-mlx-4bit"))
        XCTAssertEqual(manifest.components?.transformer, .local(path: "transformer-packed-mflux"))
        XCTAssertEqual(manifest.upstreamRepoId, "prism-ml/bonsai-image-binary-4B-mlx-1bit@main")
    }

    func testFalconPerceptionTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .visionGroundFalconPerception, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "vision-ground-falcon-perception")
        XCTAssertEqual(manifest.engine, .falconPerception)
        XCTAssertEqual(manifest.family, .falcon)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .unknown)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.visionGrounding, .visionDetection, .visionSegmentation]))
        XCTAssertEqual(manifest.upstreamRepoId, "tiiuae/Falcon-Perception")
    }

    func testHiDreamO1DevTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .hidreamO1Dev, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-hidream-o1-dev")
        XCTAssertEqual(manifest.engine, .hidreamO1)
        XCTAssertEqual(manifest.family, .hidream)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 28)
        XCTAssertEqual(manifest.defaults?.cfg, 0.0)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.txt2img, .referenceEdit, .subjectPersonalization]))
        XCTAssertEqual(manifest.upstreamRepoId, "HiDream-ai/HiDream-O1-Image-Dev")
    }

    func testHiDreamO1FullTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .hidreamO1, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-hidream-o1")
        XCTAssertEqual(manifest.engine, .hidreamO1)
        XCTAssertEqual(manifest.family, .hidream)
        XCTAssertEqual(manifest.variant, .base)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 50)
        XCTAssertEqual(manifest.defaults?.cfg, 5.0)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.txt2img, .referenceEdit, .subjectPersonalization]))
        XCTAssertEqual(manifest.upstreamRepoId, "HiDream-ai/HiDream-O1-Image")
    }

    func testPrivacyFilterTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .privacyFilter, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "text-anonymize-privacy-filter")
        XCTAssertEqual(manifest.engine, .openAIPrivacyFilter)
        XCTAssertEqual(manifest.family, .privacy)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.textAnonymization]))
        XCTAssertEqual(manifest.upstreamRepoId, "openai/privacy-filter")
    }
}
