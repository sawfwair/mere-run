import XCTest
@testable import MereRunCore

final class ManagedModelSupportTests: XCTestCase {
    func testEveryManagedModelSpecHasCapabilityDescriptor() {
        for spec in ManagedModelCatalog.allSpecs {
            XCTAssertNotNil(
                ManagedModelCapabilityCatalog.descriptor(for: spec.id),
                "Missing capability descriptor for \(spec.id)"
            )
        }
    }

    func testManagedCompanionsHaveCapabilityDescriptors() throws {
        let companionIDs = Set(ManagedModelCatalog.allSpecs.flatMap(\.companionModelIDs))
        for companionID in companionIDs {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: companionID))
            XCTAssertNotNil(
                ManagedModelCapabilityCatalog.descriptor(for: spec.id),
                "Missing capability descriptor for \(spec.id)"
            )
        }
    }

    func testTerraMindFloodSupportRequiresAppleSiliconAndSixteenGB() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.visionFloodTerraMindBase.rawValue)
        )
        let appleSilicon = MereRunMachineProfile(
            physicalMemoryBytes: 16 * 1_073_741_824,
            processorName: "M2 Pro",
            isAppleSiliconMac: true
        )
        let linux = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "Linux",
            isAppleSiliconMac: false,
            isLinux: true
        )

        let supported = ManagedModelCapabilityCatalog.support(for: spec, on: appleSilicon)
        let rejected = ManagedModelCapabilityCatalog.support(for: spec, on: linux)

        XCTAssertTrue(supported.isSupported)
        XCTAssertEqual(supported.descriptor.minimumUnifiedMemoryGB, 16)
        XCTAssertEqual(supported.descriptor.recommendedUnifiedMemoryGB, 24)
        XCTAssertFalse(rejected.isSupported)
        XCTAssertEqual(rejected.reasons, ["TerraMind Flood requires Apple Silicon macOS."])
    }

    func testLargeCoderModelIsRejectedBelowMemoryThreshold() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: CodeGenResources.defaultModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M2 Max",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertFalse(report.isSupported)
        XCTAssertTrue(report.reasons.joined(separator: " ").contains("Requires at least 64 GB"))
    }

    func testLargeCoderModelIsSupportedWhenMemoryThresholdIsMet() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: CodeGenResources.defaultModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M3 Ultra",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertTrue(report.isSupported)
        XCTAssertEqual(report.reasons, [])
    }

    func testDenseGemma4IsRejectedOnThirtyTwoGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4Resources.defaultModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M1 Max",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertFalse(report.isSupported)
        XCTAssertTrue(report.reasons.joined(separator: " ").contains("Requires at least 48 GB"))
    }

    func testGemma4TurboIsSupportedOnThirtyTwoGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4Resources.turboModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M1 Max",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertTrue(report.isSupported)
        XCTAssertEqual(report.reasons, [])
    }

    func testLagunaRequiresNinetySixGBAndRecommendsOneTwentyEightGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: LagunaResources.modelID))
        let undersized = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )
        let supported = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let rejected = ManagedModelCapabilityCatalog.support(for: spec, on: undersized)
        let accepted = ManagedModelCapabilityCatalog.support(for: spec, on: supported)

        XCTAssertFalse(rejected.isSupported)
        XCTAssertTrue(rejected.reasons.joined(separator: " ").contains("Requires at least 96 GB"))
        XCTAssertTrue(accepted.isSupported)
        XCTAssertTrue(accepted.meetsRecommendedMemory)
        XCTAssertEqual(accepted.descriptor.recommendedUnifiedMemoryGB, 128)
    }

    func testLagunaXSRequiresThirtySixGBAndRecommendsFortyEightGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: LagunaResources.xsModelID))
        let undersized = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )
        let supported = MereRunMachineProfile(
            physicalMemoryBytes: 36 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let rejected = ManagedModelCapabilityCatalog.support(for: spec, on: undersized)
        let accepted = ManagedModelCapabilityCatalog.support(for: spec, on: supported)

        XCTAssertFalse(rejected.isSupported)
        XCTAssertTrue(accepted.isSupported)
        XCTAssertEqual(accepted.descriptor.minimumUnifiedMemoryGB, 36)
        XCTAssertEqual(accepted.descriptor.recommendedUnifiedMemoryGB, 48)
    }

    func testInklingSmallRequiresOneTwentyEightGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: InklingResources.modelID))
        let undersized = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )
        let supported = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let rejected = ManagedModelCapabilityCatalog.support(for: spec, on: undersized)
        let accepted = ManagedModelCapabilityCatalog.support(for: spec, on: supported)

        XCTAssertFalse(rejected.isSupported)
        XCTAssertTrue(
            rejected.reasons.joined(separator: " ").contains("Requires at least 128 GB")
        )
        XCTAssertTrue(accepted.isSupported)
        XCTAssertTrue(accepted.meetsRecommendedMemory)
        XCTAssertEqual(accepted.descriptor.minimumUnifiedMemoryGB, 128)
        XCTAssertEqual(accepted.descriptor.recommendedUnifiedMemoryGB, 128)
    }

    func testMuseGlimmerQ4RequiresThirtyTwoGBAndRecommendsSixtyFourGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MuseGlimmerResources.modelId))
        let undersized = MereRunMachineProfile(
            physicalMemoryBytes: 16 * 1_073_741_824,
            processorName: "M4 Pro",
            isAppleSiliconMac: true
        )
        let supported = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let rejected = ManagedModelCapabilityCatalog.support(for: spec, on: undersized)
        let accepted = ManagedModelCapabilityCatalog.support(for: spec, on: supported)

        XCTAssertFalse(rejected.isSupported)
        XCTAssertTrue(rejected.reasons.joined(separator: " ").contains("Requires at least 32 GB"))
        XCTAssertTrue(accepted.isSupported)
        XCTAssertTrue(accepted.meetsRecommendedMemory)
        XCTAssertEqual(accepted.descriptor.minimumUnifiedMemoryGB, 32)
        XCTAssertEqual(accepted.descriptor.recommendedUnifiedMemoryGB, 64)
    }

    func testQ36NanoIsSupportedOnThirtyTwoGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.q36NanoModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M2 Max",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertTrue(report.isSupported)
        XCTAssertEqual(report.reasons, [])
    }

    func testQ38TwentySevenBRequiresHighMemoryAppleSilicon() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.q38TwentySevenBModelId))
        let undersized = MereRunMachineProfile(
            physicalMemoryBytes: 48 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )
        let recommended = MereRunMachineProfile(
            physicalMemoryBytes: 96 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let rejected = ManagedModelCapabilityCatalog.support(for: spec, on: undersized)
        let accepted = ManagedModelCapabilityCatalog.support(for: spec, on: recommended)

        XCTAssertFalse(rejected.isSupported)
        XCTAssertTrue(rejected.reasons.joined(separator: " ").contains("Requires at least 64 GB"))
        XCTAssertTrue(accepted.isSupported)
        XCTAssertTrue(accepted.meetsRecommendedMemory)
        XCTAssertEqual(accepted.descriptor.minimumUnifiedMemoryGB, 64)
        XCTAssertEqual(accepted.descriptor.recommendedUnifiedMemoryGB, 96)
    }

    func testQ38TwentySevenB4BitSupportsThirtyTwoGBAndRecommendsFortyEightGB() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Q35Resources.q38TwentySevenB4BitModelId)
        )
        let minimum = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M4 Pro",
            isAppleSiliconMac: true
        )
        let recommended = MereRunMachineProfile(
            physicalMemoryBytes: 48 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let minimumReport = ManagedModelCapabilityCatalog.support(for: spec, on: minimum)
        let recommendedReport = ManagedModelCapabilityCatalog.support(for: spec, on: recommended)

        XCTAssertTrue(minimumReport.isSupported)
        XCTAssertFalse(minimumReport.meetsRecommendedMemory)
        XCTAssertTrue(recommendedReport.isSupported)
        XCTAssertTrue(recommendedReport.meetsRecommendedMemory)
        XCTAssertEqual(recommendedReport.descriptor.minimumUnifiedMemoryGB, 32)
        XCTAssertEqual(recommendedReport.descriptor.recommendedUnifiedMemoryGB, 48)
    }

    func testBonsai27BIsSupportedOnSixteenGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.bonsai27B1BitModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 16 * 1_073_741_824,
            processorName: "M4 Pro",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertTrue(report.isSupported)
        XCTAssertTrue(report.meetsRecommendedMemory)
        XCTAssertEqual(report.descriptor.minimumUnifiedMemoryGB, 12)
        XCTAssertEqual(report.descriptor.recommendedUnifiedMemoryGB, 16)
    }

    func testTernaryBonsai27BIsSupportedOnTwentyFourGB() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.bonsai27B2BitModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 24 * 1_073_741_824,
            processorName: "M4 Pro",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertTrue(report.isSupported)
        XCTAssertTrue(report.meetsRecommendedMemory)
        XCTAssertEqual(report.descriptor.minimumUnifiedMemoryGB, 16)
        XCTAssertEqual(report.descriptor.recommendedUnifiedMemoryGB, 24)
    }

    func testDenseLFM25RunsInCompactMemoryTier() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: LFM2Resources.denseModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 12 * 1_073_741_824,
            processorName: "M2",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertTrue(report.isSupported)
        XCTAssertTrue(report.meetsRecommendedMemory)
        XCTAssertEqual(report.descriptor.minimumUnifiedMemoryGB, 8)
        XCTAssertEqual(report.descriptor.recommendedUnifiedMemoryGB, 12)
    }

    func testLFM25VisionRunsInCompactMemoryTier() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: LFM2Resources.visionModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 16 * 1_073_741_824,
            processorName: "M2",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertTrue(report.isSupported)
        XCTAssertTrue(report.meetsRecommendedMemory)
        XCTAssertEqual(report.descriptor.minimumUnifiedMemoryGB, 8)
        XCTAssertEqual(report.descriptor.recommendedUnifiedMemoryGB, 16)
    }


    func testNorthMiniCodeIsSupportedOnThirtyTwoGBButNotPiStartable() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: NorthMiniCodeResources.modelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M4 Pro",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog
                .allTierRecommendations(on: machine)
                .first { $0.id == NorthMiniCodeResources.modelId }
        )

        XCTAssertTrue(report.isSupported)
        XCTAssertFalse(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .textCode)
    }

    func testRecommendedCodeModelUsesNorthMiniOnThirtyTwoGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M4",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(ManagedModelCapabilityCatalog.recommendedCodeModelReport(on: machine))

        XCTAssertEqual(recommendation.spec.id, NorthMiniCodeResources.modelId)
        XCTAssertTrue(recommendation.isSupported)
    }

    func testRecommendedCodeModelUsesQwen3CoderOnSixtyFourGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(ManagedModelCapabilityCatalog.recommendedCodeModelReport(on: machine))

        XCTAssertEqual(recommendation.spec.id, CodeGenResources.defaultModelId)
        XCTAssertTrue(recommendation.isSupported)
    }

    func testSupportedCodeBenchmarkModelsExcludeQwen3BelowSixtyFourGB() {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M4",
            isAppleSiliconMac: true
        )

        let modelIDs = ManagedModelCapabilityCatalog.supportedCodeBenchmarkModelIDs(on: machine)

        XCTAssertEqual(modelIDs, [
            Q35Resources.ornith9BModelId,
            NorthMiniCodeResources.modelId,
        ])
    }

    func testOrnith9BIsSupportedOnTwentyFourGBAndStartableThroughQ35() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.ornith9BModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 24 * 1_073_741_824,
            processorName: "M4 Pro",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog
                .allTierRecommendations(on: machine)
                .first { $0.id == Q35Resources.ornith9BModelId }
        )

        XCTAssertTrue(report.isSupported)
        XCTAssertTrue(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .textChatQ35)
    }

    func testOrnith35BIsSupportedOnSixtyFourGBButNotPiStartable() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Ornith35BCodeResources.modelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog
                .allTierRecommendations(on: machine)
                .first { $0.id == Ornith35BCodeResources.modelId }
        )

        XCTAssertTrue(report.isSupported)
        XCTAssertFalse(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .textCode)
    }

    func testOrnith35BMLXIsSupportedOnOneHundredTwentyEightGBAndStartableThroughQ35() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.ornith35BMLXModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog
                .allTierRecommendations(on: machine)
                .first { $0.id == Q35Resources.ornith35BMLXModelId }
        )

        XCTAssertTrue(report.isSupported)
        XCTAssertTrue(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .textChatQ35)
        XCTAssertEqual(recommendation.minimumUnifiedMemoryGB, 96)
        XCTAssertEqual(recommendation.recommendedUnifiedMemoryGB, 128)
    }

    func testOrnith35BMLXRejectsSixtyFourGBMachine() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.ornith35BMLXModelId))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertFalse(report.isSupported)
        XCTAssertTrue(report.reasons.contains { $0.contains("96 GB") })
    }

    func testUnsupportedRuntimeIsRejected() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-klein-nano"))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "Intel",
            isAppleSiliconMac: false
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertFalse(report.isSupported)
        XCTAssertEqual(report.reasons, ["Apple Silicon macOS or Linux is required."])
    }

    func testLinuxRuntimeIsSupportedWhenMemoryThresholdIsMet() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-klein-nano"))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "Linux",
            isAppleSiliconMac: false,
            isLinux: true
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertTrue(report.isSupported)
        XCTAssertEqual(report.reasons, [])
    }

    func testRecommendedSetupOnlyContainsSupportedModels() {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 16 * 1_073_741_824,
            processorName: "M1",
            isAppleSiliconMac: true
        )

        let reports = ManagedModelCapabilityCatalog.recommendedSetupReports(on: machine)

        XCTAssertFalse(reports.isEmpty)
        XCTAssertTrue(reports.allSatisfy(\.isSupported))
        XCTAssertFalse(reports.contains { $0.spec.id == CodeGenResources.defaultModelId })
    }

    func testChatBandRecommendationsNameOneWinnerPerRAMBand() {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let bands = ManagedModelCapabilityCatalog.recommendedChatBandReports(on: machine)

        XCTAssertEqual(bands.map(\.bandLabel), ["16-23 GB", "24-63 GB", "64-95 GB", "96+ GB"])
        XCTAssertEqual(bands.map(\.modelID), [
            Gemma4Resources.twelveB4BitModelId,
            Gemma4Resources.twelveB4BitModelId,
            Gemma4Resources.twelveB4BitModelId,
            DeepseekV4FlashResources.defaultModelId,
        ])
        XCTAssertTrue(bands.allSatisfy { !$0.title.isEmpty && !$0.summary.isEmpty })
    }

    func testChatBandRecommendationDetectsCurrentMachineBand() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M2 Max",
            isAppleSiliconMac: true
        )

        let currentBand = try XCTUnwrap(
            ManagedModelCapabilityCatalog
                .recommendedChatBandReports(on: machine)
                .first { $0.contains(unifiedMemoryGB: machine.unifiedMemoryGB) }
        )

        XCTAssertEqual(currentBand.modelID, Gemma4Resources.twelveB4BitModelId)
    }

    func testChatBandRecommendationUsesGGUFQ36OnLinux() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "Linux",
            isAppleSiliconMac: false,
            isLinux: true
        )

        let currentBand = try XCTUnwrap(
            ManagedModelCapabilityCatalog
                .recommendedChatBandReports(on: machine)
                .first { $0.contains(unifiedMemoryGB: machine.unifiedMemoryGB) }
        )

        XCTAssertEqual(currentBand.modelID, ModelResolver.ModelID.q36NanoGGUF.rawValue)
    }

    func testRecommendedSetupIncludesDeepseekV4FlashOnLargeAppleSilicon() {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let reports = ManagedModelCapabilityCatalog.recommendedSetupReports(on: machine)

        XCTAssertTrue(reports.contains { $0.spec.id == DeepseekV4FlashResources.defaultModelId })
    }

    func testAgentTierSelectsNineBOnSixteenGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 16 * 1_073_741_824,
            processorName: "M1",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))

        XCTAssertEqual(recommendation.id, Q35Resources.ornith9BModelId)
        XCTAssertTrue(recommendation.isStartableByMereRun)
    }

    func testAgentTierSelectsGemmaOnThirtyTwoGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M2 Max",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))

        XCTAssertEqual(recommendation.id, Gemma4Resources.twelveB4BitModelId)
        XCTAssertEqual(recommendation.servingEngine, .textChatGemma4)
        XCTAssertTrue(recommendation.isStartableByMereRun)
    }

    func testAgentTierSelectsGemmaOnSixtyFourGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M3 Ultra",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))

        XCTAssertEqual(recommendation.id, Gemma4Resources.twelveB4BitModelId)
        XCTAssertEqual(recommendation.servingEngine, .textChatGemma4)
        XCTAssertTrue(recommendation.isStartableByMereRun)
    }

    func testAgentPremierSelectsDeepseekV4FlashOnNinetySixGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 96 * 1_073_741_824,
            processorName: "M3 Ultra",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .premier, on: machine))

        XCTAssertEqual(recommendation.id, DeepseekV4FlashResources.defaultModelId)
        XCTAssertTrue(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .deepseekV4Flash)
    }

    func testAgentPremierSelectsDeepseekV4FlashOnOneTwentyEightGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "M3 Ultra",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .premier, on: machine))

        XCTAssertEqual(recommendation.id, DeepseekV4FlashResources.defaultModelId)
        XCTAssertTrue(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .deepseekV4Flash)
    }

    func testAgentTierSelectsDeepseekV4FlashOnNinetySixGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 96 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))

        XCTAssertEqual(recommendation.id, DeepseekV4FlashResources.defaultModelId)
        XCTAssertTrue(recommendation.isStartableByMereRun)
    }

    func testTierRecommendationsExcludeRemovedQ35Models() {
        let recommendations = MereRunAgentModelCatalog.allTierRecommendations(
            on: MereRunMachineProfile(
                physicalMemoryBytes: 128 * 1_073_741_824,
                processorName: "M4 Max",
                isAppleSiliconMac: true
            )
        )

        XCTAssertFalse(recommendations.contains { $0.id == "text-chat-q35" })
        XCTAssertFalse(recommendations.contains { $0.id == "text-chat-q35-nano" })
        XCTAssertTrue(recommendations.contains { $0.id == Q35Resources.q36NanoModelId })
        XCTAssertTrue(recommendations.contains { $0.id == Q35Resources.q38TwentySevenBModelId })
        XCTAssertTrue(recommendations.contains { $0.id == Q35Resources.q38TwentySevenB4BitModelId })
    }

    func testAgentRecommendationRejectsNonAppleSilicon() {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "Intel",
            isAppleSiliconMac: false
        )

        XCTAssertNil(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))
    }

    func testAgentTierSelectsToolCapableQ36OnLinuxWithEnoughMemory() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "Linux",
            isAppleSiliconMac: false,
            isLinux: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))

        XCTAssertEqual(recommendation.id, Q35Resources.q36NanoModelId)
        XCTAssertTrue(recommendation.isStartableByMereRun)
    }
}
