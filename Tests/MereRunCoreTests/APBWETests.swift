import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class APBWETests: MereRunCoreTestCase {
    func testBundledConfigurationMatchesPinnedProfile() throws {
        let configuration = try APBWEResources.loadBundledConfiguration()
        XCTAssertEqual(configuration.channels, 512)
        XCTAssertEqual(configuration.layers, 8)
        XCTAssertEqual(configuration.frequencyBins, 513)
        XCTAssertEqual(configuration.nFFT, 1_024)
        XCTAssertEqual(configuration.hopSize, 80)
        XCTAssertEqual(configuration.winSize, 320)
        XCTAssertEqual(configuration.lowSampleRate, 16_000)
        XCTAssertEqual(configuration.highSampleRate, 48_000)
        XCTAssertEqual(configuration.chunkSize, 96_000)
        XCTAssertEqual(configuration.overlap, 2)
    }

    func testNativeGraphHasExactUpstreamInventory() throws {
        let model = APBWEModel(
            configuration: try APBWEResources.loadBundledConfiguration()
        )
        let flattened = model.parameters().flattened()
        let parameters = Dictionary(uniqueKeysWithValues: flattened)
        XCTAssertEqual(parameters.count, APBWEResources.expectedTensorCount)
        XCTAssertEqual(
            parameters.values.reduce(0) { $0 + $1.size },
            APBWEResources.expectedScalarCount
        )
        XCTAssertEqual(parameters["conv_pre_mag.weight"]?.shape, [512, 7, 513])
        XCTAssertEqual(parameters["convnext_mag.0.dwconv.weight"]?.shape, [512, 7, 1])
        XCTAssertEqual(parameters["convnext_pha.7.pwconv1.weight"]?.shape, [1_536, 512])
        XCTAssertEqual(parameters["linear_post_pha_i.weight"]?.shape, [513, 512])
    }

    func testBandlimitedUpsamplerProducesThreeSamplesPerInputSample() {
        let source: [Float] = [0.25, -0.5, 0.75, -1]
        let upsampled = APBWEAudio.upsample16kTo48k(source)
        XCTAssertEqual(upsampled.count, source.count * 3)
        for index in source.indices {
            XCTAssertEqual(upsampled[index * 3], source[index], accuracy: 1e-5)
        }
        XCTAssertTrue(upsampled.allSatisfy(\.isFinite))
    }

    func testReflectionPaddingSupportsInputsShorterThanChunkBorder() {
        let source: [Float] = [1, 2, 3, 4]
        let padded = APBWEAudio.reflectPad(source, amount: 6)
        XCTAssertEqual(padded.count, 16)
        XCTAssertEqual(Array(padded[6..<10]), source)
        XCTAssertEqual(Array(padded.prefix(6)), [1, 2, 3, 4, 3, 2])
        XCTAssertEqual(Array(padded.suffix(6)), [3, 2, 1, 2, 3, 4])
    }

    func testPinnedOfficialCheckpointInventoryWhenFixtureIsAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_APBWE_CHECKPOINT"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_APBWE_CHECKPOINT to the pinned g_16kto48k archive."
        )
        let archive = try PyTorchStateDictArchive(
            url: URL(fileURLWithPath: path),
            verifyEntryChecksums: false
        )
        XCTAssertEqual(archive.tensors.count, APBWEResources.expectedTensorCount)
        XCTAssertEqual(
            archive.tensors.reduce(0) { $0 + $1.elementCount },
            APBWEResources.expectedScalarCount
        )
        XCTAssertEqual(Set(archive.tensors.map(\.dataType)), [.float32])
        XCTAssertEqual(
            archive.descriptor(named: "conv_pre_mag.weight")?.shape,
            [512, 513, 7]
        )
        XCTAssertEqual(
            archive.descriptor(named: "convnext_mag.0.dwconv.weight")?.shape,
            [512, 1, 7]
        )
        XCTAssertEqual(
            archive.descriptor(named: "linear_post_pha_i.weight")?.shape,
            [513, 512]
        )
        try APBWEModel(
            configuration: APBWEResources.loadBundledConfiguration()
        ).validateCheckpoint(archive)
    }

    func testManagedCatalogPinsExactMITSnapshot() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.apBWE16kTo48k.rawValue)
        )
        XCTAssertEqual(spec.category, .audio)
        XCTAssertEqual(spec.validationKind, .apBWE)
        XCTAssertEqual(spec.upstreamRepoId, APBWEResources.sourceRepository)
        XCTAssertEqual(spec.upstreamRevision, APBWEResources.sourceRevision)
        XCTAssertEqual(spec.hubFallback?.repoId, APBWEResources.artifactRepository)
        XCTAssertEqual(spec.hubFallback?.revision, APBWEResources.artifactRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, APBWEResources.pins.map(\.filename))
        XCTAssertNil(spec.usageRestriction)
        XCTAssertEqual(spec.defaultCLICommands, ["audio enhance"])
    }

    func testManifestTemplateDeclaresAudioEnhancement() {
        let manifest = MereRunModelManifest.template(
            for: .apBWE16kTo48k,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.engine, .apBWE)
        XCTAssertEqual(manifest.family, .audio)
        XCTAssertEqual(manifest.precision, .fp32)
        XCTAssertEqual(manifest.supports, [.audioEnhancement])
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(APBWEResources.sourceRepository)@\(APBWEResources.sourceRevision)"
        )
    }
}
