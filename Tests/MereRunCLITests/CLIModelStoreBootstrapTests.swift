import Foundation
import XCTest
import MereRunCore
@testable import MereRunCLI

final class CLIModelStoreBootstrapTests: XCTestCase {
    private var defaultsSuites: [String] = []
    private let legacyModelsDirEnvironmentKey = ["ZE", "RO", "MODELS", "DIR"].joined(separator: "_")

    override func tearDown() {
        for suite in defaultsSuites {
            UserDefaults.standard.removePersistentDomain(forName: suite)
        }
        defaultsSuites.removeAll()
        super.tearDown()
    }

    func testModelsRootArgumentParsesFlagForms() {
        XCTAssertEqual(
            CLIModelStoreBootstrap.modelsRootArgument(
                in: ["mere.run", "--models-root", "/Volumes/Models", "model", "list"]
            ),
            "/Volumes/Models"
        )

        XCTAssertEqual(
            CLIModelStoreBootstrap.modelsRootArgument(
                in: ["mere.run", "model", "list", "--models-root=/Volumes/Models"]
            ),
            "/Volumes/Models"
        )
    }

    func testResolvedOverridePathUsesEnvironmentWhenNoFlag() {
        let path = CLIModelStoreBootstrap.resolvedOverridePath(
            arguments: ["mere.run", "model", "list"],
            environment: [MereRunModelPaths.modelsDirEnvironmentKey: "/env/models"],
            defaults: makeDefaults()
        )

        XCTAssertEqual(path, "/env/models")
    }

    func testResolvedOverridePathUsesPersistedPathWhenNoFlagOrEnvironment() {
        let defaults = makeDefaults()
        defaults.set("/persisted/models", forKey: MereRunModelPaths.modelStorageActivePathDefaultsKey)

        let path = CLIModelStoreBootstrap.resolvedOverridePath(
            arguments: ["mere.run", "model", "list"],
            environment: [:],
            defaults: defaults
        )

        XCTAssertEqual(path, "/persisted/models")
    }

    func testResolvedOverridePathIgnoresLegacyEnvironmentVariable() {
        let path = CLIModelStoreBootstrap.resolvedOverridePath(
            arguments: ["mere.run", "model", "list"],
            environment: [legacyModelsDirEnvironmentKey: "/legacy/models"],
            defaults: makeDefaults()
        )

        XCTAssertNil(path)
    }

    func testMereRunCLIExposesGlobalModelsRootOptionInHelp() {
        let help = MereRunCLI.helpMessage(for: MereRunCLI.self, includeHidden: true)
        XCTAssertTrue(help.contains("--models-root <models-root>"))
    }

    func testMereRunCLIExposesReleaseVersion() {
        XCTAssertEqual(MereRunCLIVersion.current, "0.24.0")
        XCTAssertEqual(MereRunCLI.configuration.version, MereRunCLIVersion.current)
    }

    private func makeDefaults() -> UserDefaults {
        let suite = "CLIModelStoreBootstrapTests.\(UUID().uuidString)"
        defaultsSuites.append(suite)
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }
}
