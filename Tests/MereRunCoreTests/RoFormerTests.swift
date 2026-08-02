import MLX
@testable import MereRunCore
import XCTest

final class RoFormerTests: MereRunCoreTestCase {
    func testBundledViperXConfigurationMatchesPinnedGeometry() throws {
        let configuration = try RoFormerResources.loadBundledConfiguration()
        XCTAssertEqual(configuration.modelID, ModelResolver.ModelID.roFormerViperX1297.rawValue)
        XCTAssertEqual(configuration.sampleRate, 44_100)
        XCTAssertEqual(configuration.audioChannels, 2)
        XCTAssertEqual(configuration.chunkSize, 352_800)
        XCTAssertEqual(configuration.overlap, 2)
        XCTAssertEqual(configuration.depth, 12)
        XCTAssertEqual(configuration.frequenciesPerBand.count, 62)
        XCTAssertEqual(configuration.frequenciesPerBand.reduce(0, +), 1_025)
        XCTAssertEqual(configuration.frequencyBandInputDimensions.reduce(0, +), 4_100)
        XCTAssertTrue(configuration.zeroDC)
    }

    func testNativeModelParameterInventoryMatchesCheckpoint() throws {
        let model = BSRoFormer(configuration: try RoFormerResources.loadBundledConfiguration())
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(parameters.count, RoFormerModelProfile.viperX1297.expectedTensorCount)
        XCTAssertEqual(
            parameters.values.reduce(0) { $0 + $1.shape.reduce(1, *) },
            RoFormerModelProfile.viperX1297.expectedScalarCount
        )
        XCTAssertEqual(parameters["band_split.to_features.0.0.gamma"]?.shape, [8])
        XCTAssertEqual(parameters["layers.0.0.layers.0.0.to_qkv.weight"]?.shape, [1_536, 512])
        XCTAssertEqual(parameters["layers.11.1.layers.0.0.rotary_embed.freqs"]?.shape, [32])
        XCTAssertEqual(parameters["mask_estimators.0.to_freqs.61.0.2.weight"]?.shape, [1_032, 2_048])
        XCTAssertEqual(parameters["final_norm.gamma"]?.shape, [512])
    }

    func testBundledFourStemConfigurationAndParameterInventory() throws {
        let configuration = try RoFormerResources.loadBundledConfiguration(profile: .fourStem)
        XCTAssertEqual(configuration.modelID, ModelResolver.ModelID.roFormerFourStem.rawValue)
        XCTAssertEqual(configuration.chunkSize, 485_100)
        XCTAssertEqual(configuration.dim, 384)
        XCTAssertEqual(configuration.depth, 8)
        XCTAssertEqual(configuration.numStems, 4)
        XCTAssertEqual(configuration.stemNames, ["drums", "bass", "other", "vocals"])
        XCTAssertEqual(configuration.mlpExpansionFactor, 2)
        XCTAssertEqual(configuration.transformerExpansionFactor, 4)

        let model = BSRoFormer(configuration: configuration)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(parameters.count, RoFormerModelProfile.fourStem.expectedTensorCount)
        XCTAssertEqual(
            parameters.values.reduce(0) { $0 + $1.shape.reduce(1, *) },
            RoFormerModelProfile.fourStem.expectedScalarCount
        )
        XCTAssertEqual(parameters["layers.0.0.layers.0.0.to_qkv.weight"]?.shape, [1_536, 384])
        XCTAssertEqual(parameters["layers.0.0.layers.0.0.to_out.0.weight"]?.shape, [384, 512])
        XCTAssertEqual(parameters["mask_estimators.3.to_freqs.61.0.2.weight"]?.shape, [1_032, 768])
    }

    func testSTFTAndISTFTRoundTrip() {
        let sampleCount = 64
        let left = (0..<sampleCount).map { sin(Float($0) * 0.19) }
        let right = (0..<sampleCount).map { cos(Float($0) * 0.13) * 0.5 }
        let audio = MLXArray(left + right).reshaped(1, 2, sampleCount)
        let window = RoFormerDSP.periodicHannWindow(length: 16)
        let spectrum = RoFormerDSP.stft(audio, nFFT: 16, hopLength: 4, window: window)
        let reconstructed = RoFormerDSP.istft(
            spectrum.expandedDimensions(axis: 1),
            channels: 2,
            length: sampleCount,
            nFFT: 16,
            hopLength: 4,
            window: window,
            zeroDC: false
        )[0, 0]
        MLX.eval(reconstructed)
        let values = reconstructed.asArray(Float.self)
        XCTAssertEqual(values.count, left.count + right.count)
        XCTAssertLessThan(zip(values, left + right).map { abs($0 - $1) }.max() ?? 1, 1e-4)

        let zeroedDC = RoFormerDSP.istft(
            spectrum.expandedDimensions(axis: 1),
            channels: 2,
            length: sampleCount,
            nFFT: 16,
            hopLength: 4,
            window: window,
            zeroDC: true
        )
        MLX.eval(zeroedDC)
        XCTAssertTrue(zeroedDC.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testChunkPlanAndFadeMatchPublishedDemix() throws {
        let plan = try RoFormerChunkPlan.make(
            sampleCount: 1_000_000,
            chunkSize: 352_800,
            overlap: 2
        )
        XCTAssertEqual(plan.step, 176_400)
        XCTAssertEqual(plan.border, 176_400)
        XCTAssertEqual(plan.fadeSize, 35_280)
        XCTAssertTrue(plan.usesBorderPadding)
        XCTAssertEqual(plan.starts.first, 0)
        XCTAssertEqual(plan.starts.last, 1_234_800)
        let first = plan.fadeWindow(chunkSize: 352_800, chunkIndex: 0)
        XCTAssertEqual(first[0], 1)
        XCTAssertLessThan(first.last ?? 1, 0.001)
        let last = plan.fadeWindow(chunkSize: 352_800, chunkIndex: plan.starts.count - 1)
        XCTAssertEqual(last.last, 1)
        XCTAssertEqual(last[0], 0)
    }

    func testStereoInterleaveAndReflectionHelpers() {
        let interleaved: [Float] = [1, 10, 2, 20, 3, 30, 4, 40]
        let channels = RoFormerSeparator.deinterleave(interleaved, channels: 2)
        XCTAssertEqual(channels, [[1, 2, 3, 4], [10, 20, 30, 40]])
        XCTAssertEqual(RoFormerSeparator.interleave(channels), interleaved)
        XCTAssertEqual(RoFormerSeparator.reflectPad([1, 2, 3, 4], amount: 2), [3, 2, 1, 2, 3, 4, 3, 2])
    }

    func testManagedCatalogPinsMITViperXSnapshot() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.roFormerViperX1297.rawValue)
        )
        XCTAssertEqual(spec.category, .music)
        XCTAssertEqual(spec.validationKind, .roFormer)
        XCTAssertEqual(spec.upstreamRepoId, RoFormerResources.repository)
        XCTAssertEqual(spec.upstreamRevision, RoFormerResources.revision)
        XCTAssertNil(spec.usageRestriction)
        XCTAssertEqual(spec.defaultCLICommands, ["music separate"])
        XCTAssertEqual(spec.hubFallback?.revision, RoFormerResources.revision)
    }

    func testManagedCatalogPinsMITFourStemSnapshot() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.roFormerFourStem.rawValue)
        )
        XCTAssertEqual(spec.category, .music)
        XCTAssertEqual(spec.validationKind, .roFormer)
        XCTAssertEqual(spec.upstreamRepoId, RoFormerResources.repository)
        XCTAssertEqual(spec.upstreamRevision, RoFormerResources.revision)
        XCTAssertNil(spec.usageRestriction)
        XCTAssertEqual(spec.defaultCLICommands, ["music separate"])
        XCTAssertEqual(
            spec.hubFallback?.patterns,
            [
                "LICENSE",
                "README.md",
                "bs_roformer/multistem/config.yaml",
                "bs_roformer/multistem/model.safetensors",
            ]
        )
    }
}
