import Foundation
import StudioKit

/// One `start` call captured by `RecordingProcessRunner`, with the callbacks a test drives to
/// simulate the child's stdout, stderr and exit.
package struct RecordingStart {
    package let configuration: MereRunProcessConfiguration
    package let stdout: @Sendable (String) -> Void
    package let stderr: @Sendable (String) -> Void
    package let termination: @Sendable (Int32) -> Void
}

/// Test double for `MereRunProcessRunning`: records every launch instead of spawning a process.
package final class RecordingProcessRunner: MereRunProcessRunning {
    package init() {}

    package private(set) var starts: [RecordingStart] = []
    package private(set) var processes: [RecordingProcess] = []
    /// When set, the next `start` throws instead of launching, simulating a missing executable.
    package var launchError: Error?

    package func start(
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

package final class RecordingProcess: MereRunRunningProcess {
    package private(set) var terminateCallCount = 0
    package private(set) var interruptCallCount = 0
    package private(set) var standardInputs: [String] = []
    private let acceptsStandardInput: Bool

    package init(acceptsStandardInput: Bool = true) {
        self.acceptsStandardInput = acceptsStandardInput
    }

    package func terminate() {
        terminateCallCount += 1
    }

    package func interrupt() {
        interruptCallCount += 1
    }

    package func sendStandardInput(_ text: String) throws {
        guard acceptsStandardInput else {
            throw MereRunProcessInputError.unavailable
        }
        standardInputs.append(text)
    }
}

package final class StubFileProbe: MereRunFileProbing {
    package init() {}

    package var existingPaths: Set<String> = []
    package func fileExists(atPath path: String) -> Bool { existingPaths.contains(path) }
}

package struct StubLaunchError: LocalizedError {
    package init() {}

    package var errorDescription: String? { "mere.run could not be launched." }
}
