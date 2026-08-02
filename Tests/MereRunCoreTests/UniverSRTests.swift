import Foundation
@testable import MereRunCore
import XCTest

final class UniverSRTests: MereRunCoreTestCase {
    func testBundledConfigurationMatchesPinnedProfile() throws {
        let configuration = try UniverSRResources.loadBundledConfiguration()

        XCTAssertEqual(configuration.supportedInputRates, [8_000, 12_000, 16_000, 24_000])
        XCTAssertEqual(configuration.transform.samplingRate, 48_000)
        XCTAssertEqual(configuration.model.dimensions, [96, 192, 384, 768])
        XCTAssertEqual(configuration.model.depths, [2, 2, 4, 2])
        XCTAssertEqual(configuration.inference.odeMethod, "midpoint")
        XCTAssertEqual(configuration.inference.odeSteps, 4)
    }

    func testManagedArtifactsPinOfficialCCBY40Snapshot() {
        XCTAssertEqual(UniverSRResources.sourceRevision.count, 40)
        XCTAssertEqual(UniverSRResources.artifactRevision.count, 40)
        XCTAssertEqual(UniverSRResources.weightsPin.byteCount, 229_072_395)
        XCTAssertEqual(
            UniverSRResources.weightsPin.sha256,
            "eb99f98943cc32fa82226c2da14b32b5d890416070af4946acbce442b30dc20b"
        )
        XCTAssertEqual(UniverSRResources.pins.map(\.filename), [
            "pytorch_model.bin",
            "config.yaml",
            "README.md",
        ])
    }

    func testNativeGraphHasExactOfficialInventory() throws {
        let model = UniverSRModel(configuration: try UniverSRResources.loadBundledConfiguration())
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(parameters.count, UniverSRResources.expectedTensorCount)
        XCTAssertEqual(
            parameters.values.reduce(0) { $0 + $1.size },
            UniverSRResources.expectedScalarCount
        )
    }

    func testIntegerBandlimitedUpsamplingSupportsEveryPublishedRate() throws {
        for rate in [8_000, 12_000, 16_000, 24_000] {
            let result = try UniverSRAudio.upsampleTo48k([0, 1, 0, -1], inputRate: rate)
            XCTAssertEqual(result.count, 4 * 48_000 / rate)
            XCTAssertTrue(result.allSatisfy(\.isFinite))
        }
    }

    func testPinnedOfficialCheckpointInventoryWhenFixtureIsAvailable() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_TEST_UNIVERSR_CHECKPOINT"] else {
            throw XCTSkip("Set MERERUN_TEST_UNIVERSR_CHECKPOINT to the pinned pytorch_model.bin fixture.")
        }
        let archive = try PyTorchStateDictArchive(url: URL(fileURLWithPath: path))
        XCTAssertEqual(archive.tensors.count, UniverSRResources.expectedTensorCount)
        XCTAssertEqual(
            archive.tensors.reduce(0) { $0 + $1.elementCount },
            UniverSRResources.expectedScalarCount
        )
        XCTAssertTrue(archive.tensors.allSatisfy { $0.dataType == .float32 })

        let model = UniverSRModel(configuration: try UniverSRResources.loadBundledConfiguration())
        try model.validateCheckpoint(archive)

        let loaded = try UniverSRModel.load(
            checkpoint: UniverSRCheckpoint(
                rootURL: URL(fileURLWithPath: path).deletingLastPathComponent(),
                weightsURL: URL(fileURLWithPath: path),
                sourceConfigurationURL: URL(fileURLWithPath: path),
                modelCardURL: URL(fileURLWithPath: path),
                configuration: try UniverSRResources.loadBundledConfiguration()
            ),
            dtype: .float16
        )
        XCTAssertEqual(loaded.parameters().flattened().count, UniverSRResources.expectedTensorCount)
    }

    func testManagedCatalogPinsOfficialMITCodeAndCCBY40Checkpoint() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.univerSRAudio.rawValue)
        )
        XCTAssertEqual(spec.category, .audio)
        XCTAssertEqual(spec.validationKind, .univerSR)
        XCTAssertEqual(spec.upstreamRepoId, UniverSRResources.sourceRepository)
        XCTAssertEqual(spec.upstreamRevision, UniverSRResources.sourceRevision)
        XCTAssertEqual(spec.hubFallback?.repoId, UniverSRResources.artifactRepository)
        XCTAssertEqual(spec.hubFallback?.revision, UniverSRResources.artifactRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, UniverSRResources.pins.map(\.filename))
        XCTAssertNil(spec.usageRestriction)
        XCTAssertEqual(spec.defaultCLICommands, ["audio enhance"])
    }

    func testManifestTemplateDeclaresGeneralAudioSuperResolution() {
        let manifest = MereRunModelManifest.template(
            for: .univerSRAudio,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.engine, .univerSR)
        XCTAssertEqual(manifest.family, .audio)
        XCTAssertEqual(manifest.precision, .fp32)
        XCTAssertEqual(manifest.defaults?.steps, 4)
        XCTAssertEqual(manifest.defaults?.cfg, 1.5)
        XCTAssertEqual(manifest.supports, [.audioEnhancement])
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(UniverSRResources.sourceRepository)@\(UniverSRResources.sourceRevision)"
        )
    }
}
