import XCTest
@testable import MereRunCLI

final class StatusCommandTests: XCTestCase {
    func testMereRunCLIExposesStatusCommand() {
        let commandNames = Set(MereRunCLI.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("status"))
    }

    func testStatusParsesDefaults() throws {
        let cmd = try Status.parse([])

        XCTAssertEqual(cmd.host, "127.0.0.1")
        XCTAssertEqual(cmd.port, 8080)
        XCTAssertNil(cmd.apiKey)
        XCTAssertEqual(cmd.timeoutSeconds, 1.0)
        XCTAssertFalse(cmd.json)
    }

    func testStatusParsesOverrides() throws {
        let cmd = try Status.parse([
            "--host", "localhost",
            "--port", "11434",
            "--api-key", "secret",
            "--timeout-seconds", "2.5",
            "--json",
        ])

        XCTAssertEqual(cmd.host, "localhost")
        XCTAssertEqual(cmd.port, 11_434)
        XCTAssertEqual(cmd.apiKey, "secret")
        XCTAssertEqual(cmd.timeoutSeconds, 2.5)
        XCTAssertTrue(cmd.json)
    }

    func testFormatterShowsLoadedAndInstalledModels() {
        let snapshot = StatusSnapshot(
            server: StatusServerSnapshot(
                url: "http://127.0.0.1:8080",
                health: "up",
                detail: nil,
                loadedModels: ["text-chat-gemma4"],
                modelsDetail: nil
            ),
            modelStore: StatusModelStoreSnapshot(
                path: "/Users/test/Library/Application Support/MereRun/models",
                source: "default",
                configuredPath: nil,
                isFallbackToDefault: false
            ),
            knownModelCount: 2,
            installedModels: [
                StatusInstalledModelSnapshot(
                    id: "text-chat-gemma4",
                    category: "text-chat",
                    size: "12 GB"
                ),
            ]
        )

        let output = StatusFormatter.text(snapshot)

        XCTAssertTrue(output.contains("server: up (http://127.0.0.1:8080)"))
        XCTAssertTrue(output.contains("loaded models: text-chat-gemma4"))
        XCTAssertTrue(output.contains("installed models: 1/2"))
        XCTAssertTrue(output.contains("text-chat-gemma4 (text-chat, 12 GB)"))
    }

    func testFormatterShowsUnavailableLoadedModels() {
        let snapshot = StatusSnapshot(
            server: StatusServerSnapshot(
                url: "http://127.0.0.1:8080",
                health: "up",
                detail: nil,
                loadedModels: [],
                modelsDetail: "requires API key"
            ),
            modelStore: StatusModelStoreSnapshot(
                path: "/Users/test/Library/Application Support/MereRun/models",
                source: "default",
                configuredPath: nil,
                isFallbackToDefault: false
            ),
            knownModelCount: 0,
            installedModels: []
        )

        let output = StatusFormatter.text(snapshot)

        XCTAssertTrue(output.contains("loaded models: unavailable (requires API key)"))
        XCTAssertTrue(output.contains("installed models: 0/0"))
        XCTAssertTrue(output.contains("    none"))
    }
}
