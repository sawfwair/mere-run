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
        try FileManager.default.createDirectory(
            at: resources.textEncoderURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        XCTAssertTrue(
            FileManager.default.createFile(
                atPath: resources.textEncoderURL.path,
                contents: Data()
            )
        )

        XCTAssertTrue(resources.validate().isEmpty)
        XCTAssertTrue(isLTX25ModelRoot(root))

        try FileManager.default.removeItem(at: resources.audioVAEURL)
        XCTAssertEqual(resources.validate(), [resources.audioVAEURL])
        XCTAssertFalse(isLTX25ModelRoot(root))
    }

    func testSnapshotPinsSelfContainedManagedDistribution() {
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
        XCTAssertEqual(
            LTX25Resources.managedRepository,
            "Sawfwair/LTX-2.5-Distilled-BF16-MLX-Q4-Text"
        )
        XCTAssertEqual(
            LTX25Resources.managedRevision,
            "9f316bfb18448bf67f716006fd78a37829223b74"
        )
        XCTAssertEqual(LTX25Resources.estimatedDownloadBytes, 53_878_648_085)
        XCTAssertTrue(LTX25Resources.snapshotPatterns.contains(LTX25Resources.transformerRelativePath))
        XCTAssertFalse(LTX25Resources.snapshotPatterns.contains(LTX25Resources.textEncoderRelativePath))
        XCTAssertTrue(LTX25Resources.snapshotPatterns.contains {
            $0.contains(LTX25TextEncoderQuantizedPack.relativeDirectory)
        })
        XCTAssertFalse(LTX25Resources.snapshotPatterns.contains {
            $0.contains("dev-transformer") || $0.contains("diffusion-video-vae")
        })
    }

    func testStandaloneLoaderResolvesOfficialAudioVAEForReferenceEncoding() {
        let root = URL(fileURLWithPath: "/tmp/ltx25-official")
        let transformer = root.appendingPathComponent(LTX25Resources.transformerRelativePath)

        XCTAssertEqual(
            ltxStandaloneAudioVAEWeightsURL(
                modelRoot: root,
                isLTX23: false,
                isLTX25: true,
                transformerURL: transformer
            ),
            LTX25Resources(rootURL: root).audioVAEURL
        )
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
