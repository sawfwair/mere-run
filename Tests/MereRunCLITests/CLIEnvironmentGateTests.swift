import Foundation
import MereRunContract
import XCTest
@testable import MereRunCLI

final class CLIEnvironmentGateTests: XCTestCase {
    func testStderrLinesWarnOnUnknownAndInvalidVariables() {
        let lines = CLIEnvironmentGate.stderrLines(
            arguments: ["mere.run", "status"],
            environment: [
                "MERERUN_INKLING2_PREFILL_CHUNKS": "256",
                "MERERUN_INKLING2_PREFILL_CHUNK": "not-an-int",
            ]
        )

        XCTAssertEqual(lines.count, 2)
        XCTAssertTrue(lines[0].hasPrefix("Warning: Unknown environment variable MERERUN_INKLING2_PREFILL_CHUNKS"))
        XCTAssertTrue(lines[0].contains("did you mean MERERUN_INKLING2_PREFILL_CHUNK?"))
        XCTAssertTrue(lines[0].hasSuffix("\n"))
        XCTAssertTrue(lines[1].contains("Invalid value 'not-an-int' for MERERUN_INKLING2_PREFILL_CHUNK"))
    }

    func testStderrLinesSuppressOnBuiltInAndQuietFlags() {
        let env = ["MERERUN_MODLES_DIR": "/tmp/models"]
        XCTAssertTrue(CLIEnvironmentGate.stderrLines(arguments: ["mere.run", "--help"], environment: env).isEmpty)
        XCTAssertTrue(CLIEnvironmentGate.stderrLines(arguments: ["mere.run", "status", "-h"], environment: env).isEmpty)
        XCTAssertTrue(CLIEnvironmentGate.stderrLines(arguments: ["mere.run", "--version"], environment: env).isEmpty)
        XCTAssertTrue(CLIEnvironmentGate.stderrLines(arguments: ["mere.run", "text", "chat", "--quiet"], environment: env).isEmpty)
        XCTAssertTrue(CLIEnvironmentGate.stderrLines(arguments: ["mere.run", "text", "chat", "-q"], environment: env).isEmpty)
        XCTAssertFalse(CLIEnvironmentGate.stderrLines(arguments: ["mere.run", "text", "chat", "--", "--quiet"], environment: env).isEmpty)
    }

    func testStderrLinesSilentWhenEnvironmentIsClean() {
        let lines = CLIEnvironmentGate.stderrLines(
            arguments: ["mere.run", "status"],
            environment: [
                "MERERUN_MODELS_DIR": "/tmp/models",
                "MERERUN_API_KEY": "sk-test",
                "MERERUN_INKLING2_PREFILL_CHUNK": "512",
            ]
        )
        XCTAssertTrue(lines.isEmpty)
    }

    func testStatusFormatterAndJSONIncludeEnvironmentAuditWhenPresent() throws {
        let audit = MereRunEnvironmentRegistry.audit(environment: [
            "MERERUN_API_KEY": "super-secret-value",
            "MERERUN_INKLING2_PREFILL_CHUNK": "512",
            "MERERUN_MODLES_DIR": "/tmp/models",
        ])
        let snapshot = StatusSnapshot(
            server: StatusServerSnapshot(
                url: "http://127.0.0.1:8080",
                health: "down",
                detail: nil,
                loadedModels: [],
                modelsDetail: nil,
                runtime: nil,
                runtimeDetail: nil
            ),
            modelStore: StatusModelStoreSnapshot(
                path: "/tmp/models",
                source: "default",
                configuredPath: nil,
                isFallbackToDefault: false
            ),
            knownModelCount: 1,
            installedModels: [],
            environment: audit
        )

        let text = StatusFormatter.text(snapshot)
        XCTAssertTrue(text.contains("environment overrides: MERERUN_API_KEY=[redacted], MERERUN_INKLING2_PREFILL_CHUNK=512"))
        XCTAssertTrue(text.contains("warning: Unknown environment variable MERERUN_MODLES_DIR (did you mean MERERUN_MODELS_DIR?)."))
        XCTAssertFalse(text.contains("super-secret-value"))

        let jsonData = try JSONEncoder().encode(snapshot)
        let jsonString = String(decoding: jsonData, as: UTF8.self)
        XCTAssertFalse(jsonString.contains("super-secret-value"))
        let decoded = try JSONDecoder().decode(StatusSnapshot.self, from: jsonData)
        XCTAssertEqual(decoded.environment, audit)
    }

    func testStatusFormatterOmitsEnvironmentSectionWhenNil() {
        let snapshot = StatusSnapshot(
            server: StatusServerSnapshot(
                url: "http://127.0.0.1:8080",
                health: "down",
                detail: nil,
                loadedModels: [],
                modelsDetail: nil,
                runtime: nil,
                runtimeDetail: nil
            ),
            modelStore: StatusModelStoreSnapshot(
                path: "/tmp/models",
                source: "default",
                configuredPath: nil,
                isFallbackToDefault: false
            ),
            knownModelCount: 0,
            installedModels: []
        )

        let text = StatusFormatter.text(snapshot)
        XCTAssertFalse(text.contains("environment overrides:"))
    }
}
