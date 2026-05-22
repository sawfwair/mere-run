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

        XCTAssertEqual(recommendation.id, AgentModelResources.qwen35NineBModelId)
        XCTAssertTrue(recommendation.isStartableByMereRun)
    }

    func testAgentTierSelectsQ35NanoOnThirtyTwoGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M2 Max",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))

        XCTAssertEqual(recommendation.id, Q35Resources.nanoModelId)
        XCTAssertTrue(recommendation.isStartableByMereRun)
    }

    func testAgentTierSelectsCoderOnSixtyFourGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M3 Ultra",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))

        XCTAssertEqual(recommendation.id, CodeGenResources.defaultModelId)
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

    func testQ35IsDescribedAsAlternativeWhenDeepseekIsAvailable() throws {
        let q35 = try XCTUnwrap(
            MereRunAgentModelCatalog.allTierRecommendations(
                on: MereRunMachineProfile(
                    physicalMemoryBytes: 128 * 1_073_741_824,
                    processorName: "M4 Max",
                    isAppleSiliconMac: true
                )
            ).first { $0.id == Q35Resources.defaultModelId }
        )

        XCTAssertTrue(q35.summary.contains("alternative"))
        XCTAssertTrue(q35.summary.contains("DeepSeek V4 Flash remains the preferred setup-agent tier"))
    }

    func testAgentRecommendationRejectsNonAppleSilicon() {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "Intel",
            isAppleSiliconMac: false
        )

        XCTAssertNil(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))
    }

    func testAgentTierSelectsCoderOnLinuxWithEnoughMemory() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "Linux",
            isAppleSiliconMac: false,
            isLinux: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))

        XCTAssertEqual(recommendation.id, CodeGenResources.defaultModelId)
        XCTAssertTrue(recommendation.isStartableByMereRun)
    }
}
