import Foundation

enum ExecutionPrivacyLane: Equatable, Sendable {
    case onDevice
    case directMachine
    case hostedRelay
}

enum ExecutionPrivacyCopy {
    static func chat(for lane: ExecutionPrivacyLane) -> String {
        switch lane {
        case .onDevice:
            "Runs entirely on this iPhone. Your words stay on this device."
        case .directMachine:
            "Sent directly to your paired machine over your network or tailnet."
        case .hostedRelay:
            "Sent through your configured relay to a fleet machine."
        }
    }

    static func create(for lane: ExecutionPrivacyLane) -> String {
        switch lane {
        case .onDevice:
            "Runs entirely on this iPhone. Prompts, inputs, and outputs stay on this device."
        case .directMachine:
            "Runs directly on your paired machine over your network or tailnet."
        case .hostedRelay:
            "Runs on your fleet. Prompts, inputs, status, and artifacts pass through your configured relay."
        }
    }
}
