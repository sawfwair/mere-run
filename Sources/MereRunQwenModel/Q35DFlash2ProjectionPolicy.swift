import Foundation

/// Approximate proposal-only arithmetic; the target still confirms every token.
package enum Q35DFlash2ProjectionPolicy {
    @TaskLocal package static var matrixFFNEnabled =
        ProcessInfo.processInfo.environment["MERERUN_Q38_DFLASH2_M5_FFN"] == "1"
            || ProcessInfo.processInfo.environment["MERERUN_Q38_SWIFT_METAL"] == "1"
}
