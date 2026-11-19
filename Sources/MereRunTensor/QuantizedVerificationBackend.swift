import Foundation

/// Arithmetic policy belongs to each projection, so a target can opt in while
/// its independent draft and other resident models retain their own policy.
package enum QuantizedVerificationBackend: String, Sendable, Hashable {
    case reference
    case m5Matrix

    package static var m5Supported: Bool {
        #if os(macOS)
        M5AffineQ4Verification.isSupported
        #else
        false
        #endif
    }

    /// Row-major matrix geometry and split count for serial and verification
    /// rows. Selecting different reductions by row count can change greedy
    /// target choices between drafting and verification. The tiled variant
    /// owns its own fixed geometry-to-partition mapping.
    func matrixSplits(inputSize: Int, outputSize: Int) -> Int? {
        guard self == .m5Matrix else { return nil }
        switch (inputSize, outputSize) {
        case (5_120, 17_408), (17_408, 5_120), (5_120, 16_384),
             (5_120, 14_336), (6_144, 5_120), (5_120, 248_320), (5_120, 98_624):
            return 1
        default:
            return nil
        }
    }
}

/// Reference requests and unsupported request types keep their policy on
/// the request's task, without mutating projections shared by concurrent chats.
package enum QuantizedVerificationContext {
    @TaskLocal package static var suppressM5 = false
    @TaskLocal package static var maximumM5Rows = 8
}

/// The qualified defaults apply only inside an eligible DFlash request. Explicit
/// environment settings also apply to diagnostic runs outside that scope.
package enum QwenDFlashDefaults {
    @TaskLocal package static var active = false

    package struct Flag: Sendable {
        private let configured: Bool?

        package init(_ key: String) {
            configured = ProcessInfo.processInfo.environment[key].map { $0 == "1" }
        }

        package var enabled: Bool { configured ?? QwenDFlashDefaults.active }
    }
}
