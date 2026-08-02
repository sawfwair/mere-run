import Foundation
import MediaIO
import MereRunCore
@testable import MereRunCLI
import XCTest

final class GateSupportTests: XCTestCase {
    func testGateParsesExplicitInstalledModelQuarantine() throws {
        let gate = try Gate.parse([
            "--all-installed",
            "--require-all",
            "--skip-model",
            "vision-ocr-lighton",
        ])

        XCTAssertTrue(gate.allInstalled)
        XCTAssertTrue(gate.requireAll)
        XCTAssertEqual(gate.skipModel, "vision-ocr-lighton")
    }

    func testGateRunnerDrainsChattyChildPipesWithoutDeadlocking() async throws {
        let runner = GateRunner(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            workDirectory: FileManager.default.temporaryDirectory
        )
        let result = try await runner.exec(
            [
                "-c",
                "yes stdout | head -c 1048576; yes stderr | head -c 1048576 >&2",
            ],
            timeout: 10
        )

        XCTAssertGreaterThan(result.stdout.utf8.count, 1_000_000)
        XCTAssertGreaterThan(result.stderr.utf8.count, 1_000_000)
    }

    func testGateRunnerDoesNotMissFastChildTermination() async throws {
        let runner = GateRunner(
            executableURL: URL(fileURLWithPath: "/usr/bin/true"),
            workDirectory: FileManager.default.temporaryDirectory
        )

        for _ in 0..<50 {
            _ = try await runner.exec([], timeout: 2)
        }
    }

    func testEveryManagedCatalogEntryHasAnExplicitInstalledModelSmokePlan() {
        let specs = ManagedModelCatalog.allSpecs
        let installedIDs = Set(specs.map(\.id))
        let missing = specs.filter {
            InstalledModelSmokePlans.plan(for: $0, installedIDs: installedIDs) == nil
        }
        let plans = specs.compactMap {
            InstalledModelSmokePlans.plan(for: $0, installedIDs: installedIDs)
        }

        XCTAssertEqual(missing.map(\.id), [])
        XCTAssertEqual(plans.count, specs.count)
        XCTAssertEqual(
            Set(plans.map(\.check.suite)),
            Set(ManagedModelCategory.allCases.map(\.rawValue))
        )
        XCTAssertEqual(
            Set(plans.map(\.check.id)),
            Set(specs.map { "installed-\($0.id)" })
        )
        XCTAssertTrue(plans.allSatisfy { !$0.check.comparesBaseline })
    }

    func testSynchformerIsCoveredOnlyByARealVideoSFXCompanionRun() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "sfx-woosh-synchformer"))
        let plan = try XCTUnwrap(InstalledModelSmokePlans.plan(
            for: spec,
            installedIDs: [
                "sfx-woosh-synchformer",
                "sfx-woosh-dvflow-8s",
            ]
        ))

        XCTAssertEqual(
            plan.check.requiredModels,
            ["sfx-woosh-synchformer", "sfx-woosh-dvflow-8s"]
        )
        XCTAssertTrue(plan.check.successDetail.contains("companion consumed by true inference"))
        XCTAssertTrue(plan.check.successDetail.contains("sfx video generate"))
    }

    func testDreamXRequiresWanAndUsesAWorldTransition() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "video-dreamx-world-5b-ar-mlx"))

        XCTAssertNil(InstalledModelSmokePlans.plan(
            for: spec,
            installedIDs: ["video-dreamx-world-5b-ar-mlx"]
        ))

        let plan = try XCTUnwrap(InstalledModelSmokePlans.plan(
            for: spec,
            installedIDs: [
                "video-dreamx-world-5b-ar-mlx",
                "video-wan22-ti2v-5b-mlx",
            ]
        ))
        XCTAssertEqual(
            plan.check.requiredModels,
            ["video-dreamx-world-5b-ar-mlx", "video-wan22-ti2v-5b-mlx"]
        )
        XCTAssertTrue(plan.check.successDetail.contains("world serve transition"))
    }

    func testVideoSuiteCoversDraftFullAVAndA2VidAsTrueGenerationChecks() {
        let checks = GateChecks.all.filter { $0.suite == "video" }

        XCTAssertEqual(
            checks.map(\.id),
            [
                "video-ltx23-draft",
                "video-ltx23-full-av",
                "video-ltx23-a2vid",
            ]
        )
        XCTAssertTrue(checks.allSatisfy { !$0.comparesBaseline })
    }

    func testCompatibleFallbackCountsAsInstalledForRequiredReleaseCheck() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("video-ltx23-a2vid-mlx", isDirectory: true),
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(GateRunner.modelInstalled("video-ltx23-full-mlx", root: root))
        XCTAssertTrue(GateRunner.modelInstalled("video-ltx23-a2vid-mlx", root: root))
        XCTAssertFalse(GateRunner.modelInstalled("video-ltx23-av-mlx", root: root))
    }

    func testA2VidFixtureIsDecodableAndNonSilent() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let url = directory.appendingPathComponent("fixture.wav")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try GateRunner.writeSineWaveFixture(to: url)
        let decoded = try MediaAudioIO.decode(url, targetSampleRate: 24_000, channels: 2)
        let peak = decoded.samples.map(abs).max() ?? 0

        XCTAssertEqual(decoded.sampleRate, 24_000)
        XCTAssertEqual(decoded.channelCount, 2)
        XCTAssertGreaterThan(decoded.samples.count, 40_000)
        XCTAssertGreaterThan(peak, 0.1)
    }

    func testMusicSeparationGateUsesProfileSpecificStemSets() throws {
        XCTAssertEqual(
            try GateRunner.expectedMusicSeparationStemNames(
                for: ModelResolver.ModelID.roFormerViperX1297.rawValue
            ),
            ["vocals", "instrumental"]
        )
        XCTAssertEqual(
            try GateRunner.expectedMusicSeparationStemNames(
                for: ModelResolver.ModelID.roFormerFourStem.rawValue
            ),
            ["drums", "bass", "other", "vocals"]
        )
        XCTAssertEqual(
            try GateRunner.expectedMusicSeparationStemNames(
                for: ModelResolver.ModelID.melRoFormerDereverb.rawValue
            ),
            ["noreverb"]
        )
        XCTAssertEqual(
            try GateRunner.expectedMusicSeparationStemNames(
                for: ModelResolver.ModelID.melRoFormerDenoise.rawValue
            ),
            ["dry"]
        )
        XCTAssertThrowsError(
            try GateRunner.expectedMusicSeparationStemNames(for: "music-separate-unknown")
        )
    }

    func testMusicSeparationGateAllowsSilentIndividualStems() {
        XCTAssertNil(
            GateRunner.musicSeparationAudibilityFailure(
                stemPeaks: [0, 0.000_01, 0.02, 0]
            )
        )
        XCTAssertEqual(
            GateRunner.musicSeparationAudibilityFailure(
                stemPeaks: [0, 0.000_01, 0.000_1, 0]
            ),
            "music separation produced no audible stems"
        )
    }
}
