import Foundation
@testable import MereRunCore
import XCTest

final class LTX25ResourcesTests: XCTestCase {
    func testOfficialDistilledLayoutRequiresEveryRuntimeComponent() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let resources = LTX25Resources(rootURL: root)
        for relativePath in LTX25Resources.requiredRelativePaths {
            let url = root.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
        }

        XCTAssertTrue(resources.validate().isEmpty)
        XCTAssertTrue(isLTX25ModelRoot(root))

        try FileManager.default.removeItem(at: resources.audioVAEURL)
        XCTAssertEqual(resources.validate(), [resources.audioVAEURL])
        XCTAssertFalse(isLTX25ModelRoot(root))
    }

    func testSnapshotPinsOfficialImmutableFiles() {
        XCTAssertEqual(LTX25Resources.sourceRepository, "Lightricks/LTX-2.5")
        XCTAssertEqual(
            LTX25Resources.sourceRevision,
            "dd53cc2cd45bbeaa3563dfb575cba3f49cf44761"
        )
        XCTAssertEqual(
            LTX25Resources.upstreamCodeRevision,
            "d151147788a9284cca791edc6ce898007e727fe6"
        )
        XCTAssertEqual(LTX25Resources.upstreamCodeRelease, "v1.2.0")
        XCTAssertEqual(LTX25Resources.estimatedDownloadBytes, 71_098_810_082)
        XCTAssertTrue(LTX25Resources.snapshotPatterns.contains(LTX25Resources.transformerRelativePath))
        XCTAssertTrue(LTX25Resources.snapshotPatterns.contains(LTX25Resources.textEncoderRelativePath))
        XCTAssertFalse(LTX25Resources.snapshotPatterns.contains {
            $0.contains("dev-transformer") || $0.contains("diffusion-video-vae")
        })
    }

    func testOfficialFullLayoutRequiresDevTransformerAndDistilledLoRA() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let resources = LTX25Resources(rootURL: root)
        for relativePath in LTX25Resources.fullRequiredRelativePaths {
            let url = root.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
        }

        XCTAssertTrue(resources.validateFull().isEmpty)
        XCTAssertTrue(isLTX25FullModelRoot(root))
        XCTAssertTrue(LTX25Resources.fullSnapshotPatterns.contains(LTX25Resources.diffusionVideoVAERelativePath))
        XCTAssertTrue(LTX25Resources.fullSnapshotPatterns.contains(LTX25Resources.temporalUpsamplerRelativePath))

        try FileManager.default.removeItem(at: resources.distilledLoRAURL)
        XCTAssertEqual(resources.validateFull(), [resources.distilledLoRAURL])
        XCTAssertFalse(isLTX25FullModelRoot(root))
    }
}
