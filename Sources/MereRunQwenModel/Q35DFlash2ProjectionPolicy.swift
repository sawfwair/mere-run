import Foundation
import MereRunEnvironment

/// Approximate proposal-only arithmetic; the target still confirms every token.
package enum Q35DFlash2ProjectionPolicy {
    @TaskLocal package static var matrixFFNEnabled =
        MereRunEnvironmentSwitch.isEnabled("MERERUN_Q38_DFLASH2_M5_FFN", default: false)
            || MereRunEnvironmentSwitch.isEnabled("MERERUN_Q38_SWIFT_METAL", default: false)
}
