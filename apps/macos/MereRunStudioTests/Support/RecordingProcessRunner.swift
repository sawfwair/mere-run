@testable import MereRunApp
import Foundation

/// One `start` call captured by `RecordingProcessRunner`, with the callbacks a test drives to
/// simulate the child's stdout, stderr and exit.
struct RecordingStart {
    let configuration: MereRunProcessConfiguration
    let stdout: @Sendable (String) -> Void
    let stderr: @Sendable (String) -> Void
    let termination: @Sendable (Int32) -> Void
}

/// Test double for `MereRunProcessRunning`: records every launch instead of spawning a process.
final class RecordingProcessRunner: MereRunProcessRunning {
    private(set) var starts: [RecordingStart] = []
    private(set) var processes: [RecordingProcess] = []
    /// When set, the next `start` throws instead of launching, simulating a missing executable.
    var launchError: Error?

    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess {
        if let launchError {
            self.launchError = nil
            throw launchError
        }
        let process = RecordingProcess(acceptsStandardInput: configuration.keepsStandardInputOpen)
        starts.append(
            RecordingStart(
                configuration: configuration,
                stdout: stdout,
                stderr: stderr,
                termination: termination
            )
        )
        processes.append(process)
        return process
    }
}

final class RecordingProcess: MereRunRunningProcess {
    private(set) var terminateCallCount = 0
    private(set) var interruptCallCount = 0
    private(set) var standardInputs: [String] = []
    private let acceptsStandardInput: Bool

    init(acceptsStandardInput: Bool = true) {
        self.acceptsStandardInput = acceptsStandardInput
    }

    func terminate() {
        terminateCallCount += 1
    }

    func interrupt() {
        interruptCallCount += 1
    }

    func sendStandardInput(_ text: String) throws {
        guard acceptsStandardInput else {
            throw MereRunProcessInputError.unavailable
        }
        standardInputs.append(text)
    }
}

final class StubFileProbe: MereRunFileProbing {
    var existingPaths: Set<String> = []
    func fileExists(atPath path: String) -> Bool { existingPaths.contains(path) }
}

struct StubLaunchError: LocalizedError {
    var errorDescription: String? { "mere.run could not be launched." }
}
