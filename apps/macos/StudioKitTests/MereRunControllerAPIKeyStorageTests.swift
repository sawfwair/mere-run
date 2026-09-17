import Foundation
import StudioKit
import StudioTestSupport
import XCTest

/// The runtime API key lives in the secret store (the login Keychain in the app), migrates out
/// of `UserDefaults` only after the store accepts it, and reaches the child through the
/// environment rather than argv.
@MainActor
final class MereRunControllerAPIKeyStorageTests: XCTestCase {
    private let legacyDefaultsKey = "mererun.app.runtimeAPIKey"
    private let secretName = MereRunController.runtimeAPIKeySecretName
    private var previousLegacyValue: String?

    override func setUp() {
        super.setUp()
        previousLegacyValue = UserDefaults.standard.string(forKey: legacyDefaultsKey)
        UserDefaults.standard.removeObject(forKey: legacyDefaultsKey)
    }

    override func tearDown() {
        if let previousLegacyValue {
            UserDefaults.standard.set(previousLegacyValue, forKey: legacyDefaultsKey)
        } else {
            UserDefaults.standard.removeObject(forKey: legacyDefaultsKey)
        }
        super.tearDown()
    }

    private func makeController(store: InMemorySecretStore) -> MereRunController {
        MereRunController(secretStore: store, processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
    }

    func testStoredKeyWinsOverLegacyDefaults() {
        UserDefaults.standard.set("legacy-key", forKey: legacyDefaultsKey)
        let store = InMemorySecretStore(secrets: [secretName: "keychain-key"])

        let controller = makeController(store: store)

        XCTAssertEqual(controller.runtimeAPIKey, "keychain-key")
        XCTAssertNil(controller.runtimeAPIKeyStorageNotice)
        XCTAssertEqual(UserDefaults.standard.string(forKey: legacyDefaultsKey), "legacy-key",
                       "A defaults value that was never migrated is left for the next launch to move")
    }

    func testLegacyDefaultsValueMigratesIntoStoreAndIsRemovedAfterTheWrite() {
        UserDefaults.standard.set("legacy-key", forKey: legacyDefaultsKey)
        let store = InMemorySecretStore()

        let controller = makeController(store: store)

        XCTAssertEqual(controller.runtimeAPIKey, "legacy-key")
        XCTAssertEqual(store.secrets[secretName], "legacy-key")
        XCTAssertNil(UserDefaults.standard.object(forKey: legacyDefaultsKey))
        XCTAssertNil(controller.runtimeAPIKeyStorageNotice)
    }

    func testMigrationWithRefusedWriteKeepsDefaultsAndValueAndReportsIt() throws {
        UserDefaults.standard.set("legacy-key", forKey: legacyDefaultsKey)
        let store = InMemorySecretStore()
        store.writeError = .accessDenied(operation: .write)

        let controller = makeController(store: store)

        XCTAssertEqual(controller.runtimeAPIKey, "legacy-key")
        XCTAssertEqual(controller.runtimeAuthorizationHeader, "Bearer legacy-key")
        XCTAssertEqual(UserDefaults.standard.string(forKey: legacyDefaultsKey), "legacy-key")
        XCTAssertTrue(store.secrets.isEmpty)
        let notice = try XCTUnwrap(controller.runtimeAPIKeyStorageNotice)
        XCTAssertTrue(notice.contains("next launch"), notice)
    }

    func testMigrationWithUnreadableStoreFallsBackToDefaultsAndReportsIt() throws {
        UserDefaults.standard.set("legacy-key", forKey: legacyDefaultsKey)
        let store = InMemorySecretStore()
        store.readError = .failed(operation: .read, status: -25293)

        let controller = makeController(store: store)

        XCTAssertEqual(controller.runtimeAPIKey, "legacy-key")
        XCTAssertEqual(UserDefaults.standard.string(forKey: legacyDefaultsKey), "legacy-key")
        XCTAssertTrue(store.secrets.isEmpty, "Nothing is written while the store cannot be read")
        XCTAssertNotNil(controller.runtimeAPIKeyStorageNotice)
    }

    func testMigrationRetriesOnTheNextLaunch() {
        UserDefaults.standard.set("legacy-key", forKey: legacyDefaultsKey)
        let store = InMemorySecretStore()
        store.writeError = .accessDenied(operation: .write)
        _ = makeController(store: store)

        store.writeError = nil
        let relaunched = makeController(store: store)

        XCTAssertEqual(relaunched.runtimeAPIKey, "legacy-key")
        XCTAssertEqual(store.secrets[secretName], "legacy-key")
        XCTAssertNil(UserDefaults.standard.object(forKey: legacyDefaultsKey))
        XCTAssertNil(relaunched.runtimeAPIKeyStorageNotice)
    }

    func testSettingAKeyWritesTheStoreAndNeverDefaults() {
        let store = InMemorySecretStore()
        let controller = makeController(store: store)

        controller.runtimeAPIKey = "new-key"

        XCTAssertEqual(store.secrets[secretName], "new-key")
        XCTAssertNil(UserDefaults.standard.object(forKey: legacyDefaultsKey))
        XCTAssertNil(controller.runtimeAPIKeyStorageNotice)

        controller.runtimeAPIKey = ""
        XCTAssertNil(store.secrets[secretName], "Clearing the field removes the Keychain item")
    }

    func testSettingAKeyWithRefusedWriteKeepsItForTheSessionAndReportsIt() throws {
        let store = InMemorySecretStore()
        let controller = makeController(store: store)
        store.writeError = .failed(operation: .write, status: -25300)

        controller.runtimeAPIKey = "session-key"

        XCTAssertEqual(controller.runtimeAPIKey, "session-key")
        XCTAssertEqual(controller.runtimeAuthorizationHeader, "Bearer session-key")
        XCTAssertTrue(store.secrets.isEmpty)
        XCTAssertNil(UserDefaults.standard.object(forKey: legacyDefaultsKey), "No fallback to defaults")
        let notice = try XCTUnwrap(controller.runtimeAPIKeyStorageNotice)
        XCTAssertTrue(notice.contains("could not be saved to the Keychain"), notice)

        store.writeError = nil
        controller.runtimeAPIKey = "session-key-2"
        XCTAssertEqual(store.secrets[secretName], "session-key-2")
        XCTAssertNil(controller.runtimeAPIKeyStorageNotice, "A later successful save clears the notice")
    }

    func testStatusProbeCarriesTheKeyInTheEnvironmentNotArgv() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(
            secretStore: InMemorySecretStore(), processRunner: runner, resolvesCLIOnInit: false
        )
        controller.cliPath = "/usr/bin/true"
        controller.runtimeAPIKey = " probe-key "

        let probe = Task { await controller.refreshServerStatus() }
        while runner.starts.isEmpty { await Task.yield() }

        let configuration = runner.starts[0].configuration
        XCTAssertFalse(configuration.arguments.contains("--api-key"))
        XCTAssertFalse(configuration.arguments.contains("probe-key"))
        XCTAssertEqual(configuration.environment["MERERUN_API_KEY"], "probe-key")
        XCTAssertFalse(controller.jobs.running(in: .probe).first?.displayCommand.contains("probe-key") ?? true)

        runner.starts[0].termination(0)
        _ = await probe.value
    }

    func testStatusProbeWithoutAKeySetsNoEnvironmentVariable() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(
            secretStore: InMemorySecretStore(), processRunner: runner, resolvesCLIOnInit: false
        )
        controller.cliPath = "/usr/bin/true"
        controller.runtimeAPIKey = ""

        let probe = Task { await controller.refreshServerStatus() }
        while runner.starts.isEmpty { await Task.yield() }

        XCTAssertNil(runner.starts[0].configuration.environment["MERERUN_API_KEY"])
        XCTAssertFalse(runner.starts[0].configuration.arguments.contains("--api-key"))

        runner.starts[0].termination(0)
        _ = await probe.value
    }
}
