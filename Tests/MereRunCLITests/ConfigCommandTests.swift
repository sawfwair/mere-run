import ArgumentParser
@testable import MereRunCLI
import XCTest

final class ConfigCommandTests: XCTestCase {
    func testSetParsesExplicitAndEnvironmentBackedValues() throws {
        let explicit = try Config.SetCmd.parse(["hf-endpoint", "https://huggingface.co"])
        XCTAssertEqual(explicit.key, "hf-endpoint")
        XCTAssertEqual(explicit.value, "https://huggingface.co")
        XCTAssertNil(explicit.fromEnvironment)

        let environment = try Config.SetCmd.parse([
            "hf-token",
            "--from-env",
            Config.SetCmd.valueEnvironmentKey,
        ])
        XCTAssertEqual(environment.key, "hf-token")
        XCTAssertNil(environment.value)
        XCTAssertEqual(environment.fromEnvironment, Config.SetCmd.valueEnvironmentKey)
    }

    func testSetResolvesOneValueSourceWithoutLeakingItIntoArguments() throws {
        XCTAssertEqual(
            try Config.SetCmd.resolvedValue(
                explicit: nil,
                environmentKey: Config.SetCmd.valueEnvironmentKey,
                environment: [Config.SetCmd.valueEnvironmentKey: "hf_secret"]
            ),
            "hf_secret"
        )
        XCTAssertThrowsError(
            try Config.SetCmd.resolvedValue(
                explicit: "hf_explicit",
                environmentKey: Config.SetCmd.valueEnvironmentKey,
                environment: [Config.SetCmd.valueEnvironmentKey: "hf_environment"]
            )
        )
        XCTAssertThrowsError(
            try Config.SetCmd.resolvedValue(
                explicit: nil,
                environmentKey: Config.SetCmd.valueEnvironmentKey,
                environment: [:]
            )
        )
    }
}
