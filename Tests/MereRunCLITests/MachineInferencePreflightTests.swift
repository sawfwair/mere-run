import XCTest
@testable import MereRunCLI

final class MachineInferencePreflightTests: XCTestCase {
    private let gibibyte = UInt64(1_073_741_824)
    private let imageArguments = ["mere.run", "image", "generate", "--prompt", "test"]

    func testResourceChecksReportAllCurrentBlockers() {
        let diagnostics = MachineInferencePreflight.diagnostics(
            arguments: imageArguments,
            host: MachineInferenceHostSnapshot(
                physicalMemoryBytes: 128 * gibibyte,
                availableMemoryBytes: 4 * gibibyte,
                memoryPressure: .critical,
                availableDiskBytes: gibibyte
            )
        )
        XCTAssertEqual(Set(diagnostics.map(\.id)), [
            "machine_insufficient_disk", "machine_insufficient_memory", "machine_critical_memory_pressure",
        ])
        XCTAssertTrue(diagnostics.allSatisfy { $0.severity == .blocker })
    }

    func testResourceChecksUseTheExecutionThresholdsAndModelClass() {
        let host = MachineInferenceHostSnapshot(
            physicalMemoryBytes: 128 * gibibyte,
            availableMemoryBytes: MachineInferenceClass.standard.minimumAvailableBytes,
            memoryPressure: .nominal,
            availableDiskBytes: MachineInferenceCoordinator.minimumDiskBytes(physicalMemoryBytes: 128 * gibibyte)
        )
        XCTAssertTrue(MachineInferencePreflight.diagnostics(arguments: imageArguments, host: host).isEmpty)
        let diagnostics = MachineInferencePreflight.diagnostics(
            arguments: ["mere.run", "text", "chat", "--model", "text-chat-deepseek-v4-flash"],
            host: host
        )
        XCTAssertEqual(diagnostics.map(\.id), ["machine_insufficient_memory"])
    }

    func testChatReportKeepsResourceAndMissingModelBlockersTogether() throws {
        let command = try TextChat.parse([
            "--model", "text-chat-gemma4-12b-4bit", "--require-installed", "--prompt", "test",
        ])
        let resources = MachineInferencePreflight.diagnostics(
            arguments: ["mere.run", "text", "chat", "--model", command.model],
            host: MachineInferenceHostSnapshot(
                physicalMemoryBytes: 8 * gibibyte, availableMemoryBytes: 4 * gibibyte,
                memoryPressure: .nominal, availableDiskBytes: gibibyte
            )
        )
        let report = command.makePreflightReport(
            modelID: command.model, installedModelPath: nil, resourceDiagnostics: resources
        )
        XCTAssertEqual(report.status, .blocked)
        XCTAssertEqual(Set(report.diagnostics.map(\.id)), [
            "machine_insufficient_disk", "machine_insufficient_memory", "text_chat_model_not_installed",
        ])
        let encoded = try StructuredRunOutput.encode(report)
        let decoded = try JSONDecoder().decode(TextChatPreflightReport.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded, report)
        let installed = command.makePreflightReport(
            modelID: command.model, installedModelPath: "/model", resourceDiagnostics: resources
        )
        XCTAssertEqual(installed.status, .blocked)
        XCTAssertTrue(installed.installed)
        XCTAssertEqual(installed.diagnostics, resources)
    }
}
