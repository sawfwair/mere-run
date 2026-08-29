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
            "cf8a174746cd14796c81ca2b54e035dc32e69bd8"
        )
        XCTAssertEqual(LTX25Resources.estimatedDownloadBytes, 53_878_517_792)
        XCTAssertFalse(LTX25Resources.snapshotPatterns.contains(LTX25Resources.transformerRelativePath))
        XCTAssertTrue(
            LTX25Resources.snapshotPatterns.contains(
                LTX25Resources.nativeDistilledTransformerRelativePath
            )
        )
        XCTAssertTrue(
            LTX25Resources.snapshotPatterns.contains(LTX25Resources.nativeConnectorRelativePath)
        )
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

    func testManagedNativeDistilledLayoutDoesNotRequireOfficialTransformer() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let resources = LTX25Resources(rootURL: root)
        for relativePath in LTX25Resources.requiredRelativePaths.dropFirst() {
            try writeEmptyFile(root.appendingPathComponent(relativePath))
        }
        try writeEmptyFile(resources.textEncoderURL)
        try writeNativePack(resources: resources, kind: .distilled)
        try writeNativePack(resources: resources, kind: .connector)

        XCTAssertFalse(FileManager.default.fileExists(atPath: resources.distilledTransformerURL.path))
        XCTAssertTrue(resources.validate().isEmpty)
        XCTAssertTrue(isLTX25ModelRoot(root))

        let connectorURL = LTX25NativeModelPack.outputURL(resources: resources, kind: .connector)
        try FileManager.default.removeItem(at: connectorURL)
        XCTAssertEqual(resources.validate(), [connectorURL])
    }

    func testManagedNativeFullLayoutDoesNotRequireOfficialTransformers() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let resources = LTX25Resources(rootURL: root)
        let transformerPaths = Set([
            LTX25Resources.devTransformerRelativePath,
            LTX25Resources.distilledTransformerRelativePath,
        ])
        for relativePath in LTX25Resources.fullRequiredRelativePaths
            where !transformerPaths.contains(relativePath)
        {
            try writeEmptyFile(root.appendingPathComponent(relativePath))
        }
        try writeNativePack(resources: resources, kind: .dev)
        try writeNativePack(resources: resources, kind: .distilled)
        try writeNativePack(resources: resources, kind: .connector)

        XCTAssertTrue(resources.validateFull().isEmpty)
        XCTAssertTrue(isLTX25FullModelRoot(root))

        let devURL = LTX25NativeModelPack.outputURL(resources: resources, kind: .dev)
        try FileManager.default.removeItem(at: devURL)
        XCTAssertEqual(resources.validateFull(), [devURL])
    }

    private func writeEmptyFile(_ url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
    }

    private func writeNativePack(
        resources: LTX25Resources,
        kind: LTX25NativeModelPackKind
    ) throws {
        let url = LTX25NativeModelPack.outputURL(resources: resources, kind: kind)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let header: [String: Any] = [
            "__metadata__": [
                "format": LTX25NativeModelPack.format,
                "kind": kind.rawValue,
                "source_revision": LTX25Resources.sourceRevision,
            ],
            "fixture.weight": [
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            ],
        ]
        var headerData = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys]
        )
        let padding = (8 - (headerData.count % 8)) % 8
        headerData.append(Data(repeating: 0x20, count: padding))
        var headerLength = UInt64(headerData.count).littleEndian
        var fileData = withUnsafeBytes(of: &headerLength) { Data($0) }
        fileData.append(headerData)
        fileData.append(contentsOf: [0, 0])
        try fileData.write(to: url, options: .atomic)
    }
}
