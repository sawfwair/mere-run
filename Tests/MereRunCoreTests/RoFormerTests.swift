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

    func testBundledMelBandConfigurationsAndFrequencyLayout() throws {
        let dereverb = try MelBandRoFormerResources.loadBundledConfiguration(profile: .dereverb)
        let denoise = try MelBandRoFormerResources.loadBundledConfiguration(profile: .denoise)
        XCTAssertEqual(dereverb.stemNames, ["noreverb"])
        XCTAssertEqual(dereverb.overlap, 2)
        XCTAssertEqual(denoise.stemNames, ["dry"])
        XCTAssertEqual(denoise.overlap, 4)
        XCTAssertEqual(dereverb.frequencyLayout.inputDimensions, [
            28, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 28, 28, 28, 36, 36,
            36, 40, 40, 44, 52, 52, 52, 60, 64, 68,
            76, 80, 80, 88, 96, 104, 112, 116, 124, 132,
            144, 156, 164, 176, 188, 200, 216, 228, 244, 264,
            284, 304, 320, 344, 372, 396, 420, 452, 488, 520,
        ])
        XCTAssertEqual(dereverb.frequencyLayout.bandsPerFrequency.min(), 1)
        XCTAssertEqual(dereverb.frequencyLayout.bandsPerFrequency.max(), 2)
    }

    func testMelBandModelParameterInventoryMatchesCheckpoint() throws {
        let configuration = try MelBandRoFormerResources.loadBundledConfiguration(profile: .dereverb)
        let model = MelBandRoFormer(configuration: configuration)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(parameters.count, MelBandRoFormerProfile.dereverb.expectedTensorCount)
        XCTAssertEqual(
            parameters.values.reduce(0) { $0 + $1.shape.reduce(1, *) },
            MelBandRoFormerProfile.dereverb.expectedScalarCount
        )
        XCTAssertEqual(parameters["band_split.to_features.0.0.gamma"]?.shape, [28])
        XCTAssertEqual(parameters["layers.0.0.norm.gamma"]?.shape, [384])
        XCTAssertEqual(parameters["layers.0.0.layers.0.0.to_out.0.weight"]?.shape, [384, 512])
        XCTAssertEqual(parameters["mask_estimators.0.to_freqs.59.0.4.weight"]?.shape, [1_040, 1_536])
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

    func testManagedCatalogPinsMITMelBandSnapshots() throws {
        let dereverb = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.melRoFormerDereverb.rawValue)
        )
        XCTAssertEqual(dereverb.category, .music)
        XCTAssertEqual(dereverb.validationKind, .roFormer)
        XCTAssertEqual(dereverb.upstreamRepoId, RoFormerResources.repository)
        XCTAssertEqual(dereverb.upstreamRevision, RoFormerResources.revision)
        XCTAssertNil(dereverb.usageRestriction)
        XCTAssertEqual(
            dereverb.hubFallback?.patterns,
            [
                "LICENSE",
                "README.md",
                "mel_band_roformer/dereverb/config.yaml",
                "mel_band_roformer/dereverb/model.safetensors",
            ]
        )

        let denoise = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.melRoFormerDenoise.rawValue)
        )
        XCTAssertEqual(denoise.validationKind, .roFormer)
        XCTAssertEqual(
            denoise.hubFallback?.patterns,
            [
                "LICENSE",
                "README.md",
                "mel_band_roformer/denoise/config.yaml",
                "mel_band_roformer/denoise/model.safetensors",
            ]
        )
    }
}
