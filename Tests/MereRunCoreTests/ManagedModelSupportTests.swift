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

    func testNonAppleSiliconMacIsRejected() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-klein-nano"))
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "Intel",
            isAppleSiliconMac: false
        )

        let report = ManagedModelCapabilityCatalog.support(for: spec, on: machine)

        XCTAssertFalse(report.isSupported)
        XCTAssertEqual(report.reasons, ["Apple Silicon macOS is required."])
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

    func testAgentPremierSelects122BMXFP4OnNinetySixGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 96 * 1_073_741_824,
            processorName: "M3 Ultra",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .premier, on: machine))

        XCTAssertEqual(recommendation.id, "text-agent-qwen35-122b-a10b-mxfp4")
        XCTAssertTrue(recommendation.sourceConfigurationRequired)
    }

    func testAgentPremierSelects122B8BitOnOneTwentyEightGB() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "M3 Ultra",
            isAppleSiliconMac: true
        )

        let recommendation = try XCTUnwrap(MereRunAgentModelCatalog.recommendation(for: .premier, on: machine))

        XCTAssertEqual(recommendation.id, "text-agent-qwen35-122b-a10b-8bit")
        XCTAssertTrue(recommendation.sourceConfigurationRequired)
    }

    func testAgentRecommendationRejectsNonAppleSilicon() {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "Intel",
            isAppleSiliconMac: false
        )

        XCTAssertNil(MereRunAgentModelCatalog.recommendation(for: .tier, on: machine))
    }
}
