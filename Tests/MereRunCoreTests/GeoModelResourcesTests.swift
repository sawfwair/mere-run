import XCTest
@testable import MereRunCore

final class GeoModelResourcesTests: XCTestCase {
    func testTerraMindFirePinsExactFP32ConversionContract() {
        let configuration = TerraMindFireConversionConfiguration(
            format: TerraMindFireResources.conversionFormat,
            modelID: TerraMindFireResources.defaultModelID,
            sourceRepository: TerraMindFireResources.sourceRepository,
            sourceRevision: TerraMindFireResources.sourceRevision,
            sourceCheckpoint: TerraMindFireResources.sourceCheckpointFilename,
            sourceCheckpointSHA256: TerraMindFireResources.sourceCheckpointSHA256,
            sourceConfigurationSHA256: TerraMindFireResources.sourceConfigurationSHA256,
            converter: "scripts/convert-terramind-fire-mlx.py@v1",
            dtype: "float32",
            tensorCount: TerraMindFireResources.tensorCount,
            scalarCount: TerraMindFireResources.scalarCount,
            tileSize: TerraMindFloodModel.tileSize,
            timestamps: TerraMindFloodModel.timestampCount
        )

        XCTAssertNoThrow(try TerraMindFireResources.validateConfiguration(configuration))
        XCTAssertEqual(TerraMindFireResources.tensorCount, 171)
        XCTAssertEqual(TerraMindFireResources.scalarCount, 168_416_386)
        XCTAssertEqual(
            TerraMindFireResources.weightsArtifact.sha256,
            "7ebd587e684285112554743a27d50de596e807746c4ea38cfab34999c0adf21a"
        )
    }

    func testEveryTESSERATierValidatesItsImmutableConversionContract() {
        XCTAssertEqual(
            TESSERAResources.allSpecs.map(\.variant),
            [.nano, .small, .medium, .large, .teacher]
        )
        for source in TESSERAResources.allSpecs {
            let configuration = TESSERAConversionConfiguration(
                format: TESSERAResources.conversionFormat,
                modelID: source.modelID,
                variant: source.variant,
                sourceRepository: source.sourceRepository,
                sourceRevision: source.sourceRevision,
                sourceCheckpoint: source.sourceCheckpointFilename,
                sourceCheckpointSHA256: source.sourceCheckpointSHA256,
                converter: "scripts/convert-tessera-v2-mlx.py@v1",
                dtype: "float32",
                tensorCount: source.tensorCount,
                scalarCount: source.scalarCount,
                architecture: source.architecture
            )

            XCTAssertNoThrow(try TESSERAResources.validateConfiguration(configuration, source: source))
            XCTAssertEqual(
                source.architecture.representationDimension,
                source.variant == .teacher ? 1_024 : 128
            )
            XCTAssertEqual(source.weightsArtifact.sha256.count, 64)
            XCTAssertEqual(source.configurationArtifact.sha256.count, 64)
        }
    }

    func testEveryOlmoEarthTierValidatesItsImmutableConversionContract() {
        XCTAssertEqual(OlmoEarthResources.allSpecs.map(\.variant), [.nano, .tiny, .small, .base])
        for source in OlmoEarthResources.allSpecs {
            let configuration = OlmoEarthConversionConfiguration(
                format: OlmoEarthResources.conversionFormat,
                modelID: source.modelID,
                variant: source.variant,
                sourceRepository: source.sourceRepository,
                sourceRevision: source.sourceRevision,
                sourceWeights: OlmoEarthResources.sourceWeightsFilename,
                sourceWeightsSHA256: source.sourceWeightsSHA256,
                sourceConfigurationSHA256: source.sourceConfigurationSHA256,
                converter: "scripts/convert-olmoearth-v12-mlx.py@v1",
                dtype: "float32",
                tensorCount: source.tensorCount,
                scalarCount: source.scalarCount,
                supportedModalities: OlmoEarthResources.supportedModalities,
                architecture: source.architecture
            )

            XCTAssertNoThrow(try OlmoEarthResources.validateConfiguration(configuration, source: source))
            XCTAssertEqual(source.architecture.positionEncoding, "rope_3d_mixed")
            XCTAssertEqual(source.weightsArtifact.sha256.count, 64)
            XCTAssertEqual(source.configurationArtifact.sha256.count, 64)
        }
    }

    func testHardwareRecommendationsReachStrongestDeployableTiers() {
        XCTAssertEqual(TESSERAResources.recommendedVariant(on: machine(gigabytes: 4)), .nano)
        XCTAssertEqual(TESSERAResources.recommendedVariant(on: machine(gigabytes: 6)), .small)
        XCTAssertEqual(TESSERAResources.recommendedVariant(on: machine(gigabytes: 8)), .medium)
        XCTAssertEqual(TESSERAResources.recommendedVariant(on: machine(gigabytes: 12)), .large)
        XCTAssertEqual(TESSERAResources.recommendedVariant(on: machine(gigabytes: 32)), .teacher)
        XCTAssertEqual(OlmoEarthResources.recommendedVariant(on: machine(gigabytes: 4)), .nano)
        XCTAssertEqual(OlmoEarthResources.recommendedVariant(on: machine(gigabytes: 8)), .tiny)
        XCTAssertEqual(OlmoEarthResources.recommendedVariant(on: machine(gigabytes: 12)), .small)
        XCTAssertEqual(OlmoEarthResources.recommendedVariant(on: machine(gigabytes: 16)), .base)
    }

    private func machine(gigabytes: UInt64) -> MereRunMachineProfile {
        MereRunMachineProfile(
            physicalMemoryBytes: gigabytes * 1_073_741_824,
            processorName: "Test",
            isAppleSiliconMac: true
        )
    }
}
